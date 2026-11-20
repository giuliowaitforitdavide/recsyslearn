"""Validation of the input tables and conversion of identifiers to dense codes.

The native core only understands integer codes: this module is the single place where
pandas objects are checked and translated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from recsyslearn import _core
from recsyslearn.errors import InvalidInputError, MissingColumnsError

# The core indexes rows with 32-bit integers.
_MAX_ROWS = 2**32 - 1
# ... and identifies users, items and groups with 32-bit codes.
_MAX_CODES = 2**32
# Largest magnitude below which every integer is a distinct float64.
_MAX_EXACT_INTEGER = 2**53

USER = "user"
ITEM = "item"
RANK = "rank"
GROUP = "group"


def check_frame(
    frame: Any, name: str, required: Sequence[str], optional: Sequence[str] = ()
) -> None:
    """Raise unless ``frame`` is a DataFrame with the ``required`` columns.

    The columns the function reads, ``required`` and ``optional``, must appear once;
    any other column is ignored, repeated or not.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MissingColumnsError(name, missing)
    used = {*required, *optional}
    repeated = [column for column in frame.columns[frame.columns.duplicated()] if column in used]
    if repeated:
        raise InvalidInputError(f"{name} has more than one column named {repeated[0]!r}")


def check_k(k: Any) -> int | None:
    """Validate a cutoff: ``None`` (whole list) or a positive integer."""
    if k is None:
        return None
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise InvalidInputError(f"k must be a positive integer or None, got {k!r}")
    # No list can be longer than the largest table the core accepts: a larger cutoff
    # means the whole list, and must still fit the native integer.
    return min(int(k), _MAX_ROWS)


def check_choice(name: str, value: Any, choices: Sequence[str]) -> str:
    """Validate a string option."""
    if value not in choices:
        expected = " or ".join(repr(choice) for choice in choices)
        raise InvalidInputError(f"{name} must be {expected}, got {value!r}")
    return value


def _kind(column: pd.Series) -> str:
    """What the identifiers of a column are: ``"integer"``, ``"string"``, ..."""
    values = column.cat.categories if isinstance(column.dtype, pd.CategoricalDtype) else column
    return pd.api.types.infer_dtype(values, skipna=True)


def _factorize(values: pd.Series, name: Any, *, sort: bool = True) -> tuple[np.ndarray, Any]:
    """Codes and distinct identifiers of a column, in the order of the identifiers.

    Sorting the identifiers makes the codes, and with them the order in which the core
    adds things up and breaks ties, independent of the order of the rows. Identifiers
    that cannot be compared with each other (e.g. numbers and strings in the same
    column) keep the order of first appearance.
    """
    try:
        try:
            codes, uniques = pd.factorize(values, sort=sort)
        except TypeError:
            codes, uniques = pd.factorize(values)
    except TypeError:
        raise InvalidInputError(f"{name} identifiers must be hashable") from None
    if len(codes) and codes.min() < 0:
        raise InvalidInputError(f"{name} identifiers must not be missing")
    if len(uniques) > _MAX_CODES:
        raise InvalidInputError(
            f"{len(uniques)} distinct {name} identifiers are more than the {_MAX_CODES} supported"
        )
    return np.ascontiguousarray(codes, dtype=np.uint32), uniques


def check_same_kind(first: pd.Series, second: pd.Series) -> None:
    """Refuse two columns that share no identifier because their types differ.

    To be called when no identifier of one column is found in the other: integers in
    one table and strings in the other would silently not match, and every metric
    would be computed as if the tables had nothing in common.
    """
    if _kind(first) != _kind(second):
        raise InvalidInputError(
            f"the {first.name} identifiers of the two tables have different types "
            f"({_kind(first)} and {_kind(second)}) and none of them match"
        )


def encode(*columns: pd.Series) -> tuple[list[np.ndarray], pd.Index]:
    """Map identifier columns to shared dense codes, in the order of the identifiers.

    Two columns that share no identifier *and* hold identifiers of different kinds are
    refused, see :func:`check_same_kind`.
    """
    # An empty column has no identifiers to add, and concatenating one of another
    # dtype is deprecated in pandas.
    filled = [column for column in columns if len(column)] or [columns[0]]
    values = filled[0] if len(filled) == 1 else pd.concat(filled, ignore_index=True)
    codes, uniques = _factorize(values, columns[0].name)
    bounds = np.cumsum([len(column) for column in columns])[:-1]
    split = np.split(codes, bounds)
    if len(columns) == 2 and all(len(part) for part in split):
        seen = np.zeros(len(uniques), dtype=bool)
        seen[split[0]] = True
        if not seen[split[1]].any():
            check_same_kind(*columns)
    return split, pd.Index(uniques, name=columns[0].name, tupleize_cols=False)


def encode_sorted(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, pd.Index]:
    """Codes of one identifier column of ``frame``, in the order of the identifiers."""
    (codes,), uniques = encode(frame[column])
    return codes, uniques


def values(frame: pd.DataFrame, column: str, name: str) -> np.ndarray | None:
    """The optional numeric ``column`` of ``frame`` as float64, if present."""
    if column not in frame.columns:
        return None
    if pd.api.types.is_integer_dtype(frame[column].dtype):
        largest, smallest = frame[column].max(), frame[column].min()
        # Compared as they are: negating the minimum would overflow an unsigned or
        # the most negative value.
        if pd.notna(largest) and (largest > _MAX_EXACT_INTEGER or smallest < -_MAX_EXACT_INTEGER):
            raise InvalidInputError(
                f"the {column!r} column of {name} holds integers beyond 2^53, which cannot "
                "be told apart once converted to floating point: rescale them"
            )
    try:
        return np.ascontiguousarray(frame[column].to_numpy(dtype=np.float64, na_value=np.nan))
    except (TypeError, ValueError):
        # `from None`: the pandas error quotes the offending cell.
        raise InvalidInputError(f"the {column!r} column of {name} must be numeric") from None


def _present(codes: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask, by code, of the ``n`` identifiers that appear in ``codes``."""
    mask = np.zeros(n, dtype=bool)
    mask[codes] = True
    return mask


class Recommendations:
    """Recommendation lists, optionally with the ground truth, in coded form.

    ``users`` and ``items`` map codes back to identifiers: those of the
    recommendations and, with a ground truth, those that only the ground truth knows.
    ``recommended_users`` and ``recommended_items`` are masks, by code, of the
    identifiers found in the recommendations. ``lists`` and ``truth`` are the native
    structures.
    """

    def __init__(self, recommendations: Any, ground_truth: Any = None):
        """Validate and encode the tables.

        The ground truth is validated and encoded whole, whichever of its rows the
        metric ends up reading: what makes a table invalid does not depend on the
        recommendations.
        """
        check_frame(recommendations, "recommendations", [USER, ITEM], [RANK])
        ranks = values(recommendations, RANK, "recommendations")

        if ground_truth is None:
            (user_codes,), self.users = encode(recommendations[USER])
            (item_codes,), self.items = encode(recommendations[ITEM])
        else:
            check_frame(ground_truth, "ground_truth", [USER, ITEM], [RANK])
            (user_codes, truth_users), self.users = encode(
                recommendations[USER], ground_truth[USER]
            )
            (item_codes, truth_items), self.items = encode(
                recommendations[ITEM], ground_truth[ITEM]
            )

        self.recommended_users = _present(user_codes, len(self.users))
        self.recommended_items = _present(item_codes, len(self.items))
        self.lists = _core.RankedLists(user_codes, item_codes, ranks, len(self.users))
        self.truth = None
        if ground_truth is not None:
            relevance = values(ground_truth, RANK, "ground_truth")
            self.truth = _core.GroundTruth(truth_users, truth_items, relevance, len(self.users))
