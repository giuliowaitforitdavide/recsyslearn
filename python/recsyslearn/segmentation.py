"""Segmentation of items and users into groups.

The segmentations return a table ``[item | user, "group"]`` that can be passed as-is to
the metrics of :mod:`recsyslearn.fairness`; :func:`popularity` returns the number the
segmentations by interactions are based on, as ``[item | user, "popularity"]``.
Thresholds should be computed on the training interactions.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from typing import Any, Literal

import numpy as np
import pandas as pd

from recsyslearn import _core
from recsyslearn._frames import (
    GROUP,
    ITEM,
    USER,
    check_choice,
    check_frame,
    encode_sorted,
    has_duplicates,
)
from recsyslearn.errors import InvalidInputError

__all__ = ["by_cumulative_interactions", "by_feature", "by_interaction_count", "popularity"]

Side = Literal["item", "user"]
_POPULARITY = "popularity"


def _check_side(interactions: Any, by: Any) -> str:
    """Validate the interactions and the side to segment, and return the side.

    The identifiers of the side being segmented are checked when they are encoded; the
    other column is checked here, so that an interaction without a user (item) is
    refused whichever side is segmented.
    """
    check_choice("by", by, (ITEM, USER))
    check_frame(interactions, "interactions", [USER, ITEM])
    other = USER if by == ITEM else ITEM
    if interactions[other].isna().any():
        raise InvalidInputError(f"{other} identifiers must not be missing")
    return by


def _segment(
    interactions: pd.DataFrame,
    proportions: Sequence[float],
    by: str,
    labels: Sequence[Any] | None,
    assign: Callable[[np.ndarray, list[float]], np.ndarray],
) -> pd.DataFrame:
    """Shared body of the two segmentations by number of interactions.

    Validates the arguments, counts the interactions of every item (user) and labels
    the group codes returned by ``assign``, the function of the core that receives the
    counts and the proportions and implements the rule.
    """
    by = _check_side(interactions, by)
    try:
        proportions = list(proportions)
    except TypeError:
        raise InvalidInputError("proportions must be a sequence of numbers") from None
    # One value per group. `float` would also convert strings of digits and booleans.
    if not all(
        isinstance(proportion, (numbers.Real, np.number))
        and not isinstance(proportion, (bool, np.bool_))
        for proportion in proportions
    ):
        raise InvalidInputError("proportions must be a sequence of numbers")
    proportions = [float(proportion) for proportion in proportions]
    if labels is None:
        labels = range(1, len(proportions) + 1)
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Iterable):
        raise InvalidInputError(
            f"labels must be a sequence of {len(proportions)} distinct values, one per proportion"
        )
    if isinstance(labels, (Set, Mapping)):
        # The first label names the most popular group: a set has no first element.
        raise InvalidInputError(
            "labels must be an ordered sequence (a list, a tuple, ...), in the order of proportions"
        )
    labels = list(labels)
    if not all(pd.api.types.is_scalar(label) for label in labels):
        raise InvalidInputError("labels must be scalar values (strings, numbers, ...)")
    labels = pd.Index(labels)
    if len(labels) != len(proportions) or labels.has_duplicates:
        raise InvalidInputError(
            f"labels must be {len(proportions)} distinct values, one per proportion"
        )
    if labels.hasnans:
        raise InvalidInputError("labels must not contain missing values")

    codes, entities = encode_sorted(interactions, by)
    counts = _core.interaction_counts(codes, len(entities))
    groups = assign(counts, proportions)
    return pd.DataFrame({by: entities, GROUP: labels.take(groups)})


def by_cumulative_interactions(
    interactions: pd.DataFrame,
    proportions: Sequence[float] = (0.8, 0.2),
    by: Side = "item",
    *,
    labels: Sequence[Any] | None = None,
) -> pd.DataFrame:
    r"""Segment items (or users) by the share of *interactions* each group accounts for.

    This is the short head / long tail segmentation: with the default proportions the
    first group is made of the most popular items that together account for 80% of the
    interactions, and the second group of all the others.

    Items are sorted by decreasing number of interactions. The first group is the
    smallest set of top items whose interactions reach a share :math:`p_1` of the
    total, the first two groups together the smallest set that reaches
    :math:`p_1 + p_2`, and so on: an item belongs to group :math:`g` when the items
    before it account for at least :math:`p_1 + \dots + p_{g-1}` of the interactions
    but less than :math:`p_1 + \dots + p_g`. The item on which a threshold is crossed
    therefore belongs to the more popular group.

    Parameters
    ----------
    interactions
        Columns ``["user", "item"]``; a ``rating`` column (explicit feedback) is
        ignored. Every row is one interaction: repeated pairs count.
    proportions
        Share of the interactions of each group, from the most to the least popular.
        They must be positive and sum to 1; e.g. ``(0.2, 0.6, 0.2)`` for head, mid
        and tail.
    by
        ``"item"`` (popularity) or ``"user"`` (activity).
    labels
        Names of the groups, in the order of ``proportions``: an ordered sequence (a
        list, a tuple, ...; not a set) of distinct scalar values (strings, numbers,
        ...), none of them missing. Defaults to ``1, 2, ...``, where 1 is the most
        popular group.

    Returns
    -------
    pandas.DataFrame
        Columns ``[by, "group"]``, one row per item (user), sorted by identifier.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table is invalid in one of the ways listed in :ref:`input-errors`; if
        ``by`` is not ``"item"`` or ``"user"``; if ``proportions`` is empty, holds
        values that are not positive finite numbers, or does not sum to 1 within
        :math:`10^{-9}`; or if ``labels`` is not one distinct, scalar, non-missing value
        per proportion.

    Notes
    -----
    The literature does not say how items with the same number of interactions are
    ordered. Ties are broken by ascending identifier, so that the result does not
    depend on the order of the rows.

    Proportions must sum to 1 within :math:`10^{-9}`, and a threshold within
    floating-point rounding error of an integer is taken to be that integer.

    References
    ----------
    * H. Abdollahpouri, R. Burke and B. Mobasher. *Managing popularity bias in
      recommender systems with personalized re-ranking*. FLAIRS 2019.
      https://arxiv.org/abs/1901.07555
    * H. Abdollahpouri et al. *User-centered evaluation of popularity bias in
      recommender systems*. UMAP 2021. https://arxiv.org/abs/2103.06364
    * Ò. Celma and P. Cano. *From hits to niches? Or how popular artists can bias
      music recommendation and discovery*. Netflix Prize workshop @ KDD 2008.
      https://mtg.upf.edu/files/publications/Celma-ACM-Netflix-KDD2008.pdf
    """
    return _segment(interactions, proportions, by, labels, _core.by_cumulative_interactions)


def by_interaction_count(
    interactions: pd.DataFrame,
    proportions: Sequence[float] = (0.2, 0.8),
    by: Side = "item",
    *,
    labels: Sequence[Any] | None = None,
) -> pd.DataFrame:
    r"""Segment items (or users) by number of interactions into shares of the *population*.

    Items are sorted by decreasing number of interactions; the first group is the top
    :math:`\lceil p_1 n \rceil` of the :math:`n` items, the first two groups together
    the top :math:`\lceil (p_1 + p_2) n \rceil`, and so on: what is rounded up are the
    cumulative boundaries, not the size of each group. With the default
    proportions the first group is the 20% most popular items; with
    ``proportions=(0.05, 0.95), by="user"`` it is the 5% most active users.

    Unlike :func:`by_cumulative_interactions`, the proportions refer to how many items
    are in each group, not to how many interactions they account for.

    Parameters
    ----------
    interactions, by, labels
        See :func:`by_cumulative_interactions`.
    proportions
        Share of the items (users) in each group, from the most to the least popular.
        They must be positive and sum to 1.

    Returns
    -------
    pandas.DataFrame
        Columns ``[by, "group"]``, one row per item (user), sorted by identifier.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table is invalid in one of the ways listed in :ref:`input-errors`; if
        ``by`` is not ``"item"`` or ``"user"``; if ``proportions`` is empty, holds
        values that are not positive finite numbers, or does not sum to 1 within
        :math:`10^{-9}`; or if ``labels`` is not one distinct, scalar, non-missing value
        per proportion.

    Notes
    -----
    Ties are broken by ascending identifier and the cumulative boundaries are rounded
    up, so the first group is never empty; neither choice is specified in the
    literature. With few items a group after the first can be empty: ``(0.3, 0.3,
    0.4)`` on 5 items gives boundaries 2 and 3, hence groups of 2, 1 and 2 items.

    Proportions must sum to 1 within :math:`10^{-9}`, and a boundary within
    floating-point rounding error of an integer is taken to be that integer (``0.1 +
    0.2`` of 10 items is 3, not 4).

    References
    ----------
    * Y. Li, H. Chen, Z. Fu, Y. Ge and Y. Zhang. *User-oriented fairness in
      recommendation*. WWW 2021. https://arxiv.org/abs/2104.10671
    * H. Abdollahpouri, M. Mansoury, R. Burke and B. Mobasher. *The unfairness of
      popularity bias in recommendation*. RMSE @ RecSys 2019.
      https://arxiv.org/abs/1907.13286
    """
    return _segment(interactions, proportions, by, labels, _core.by_interaction_count)


def popularity(
    interactions: pd.DataFrame,
    by: Side = "item",
    *,
    relative_to: Literal["interactions", "population"] = "interactions",
) -> pd.DataFrame:
    r"""Popularity of every item (or activity of every user) as a fraction.

    Parameters
    ----------
    interactions
        Columns ``["user", "item"]``; a ``rating`` column (explicit feedback) is ignored.
    by
        ``"item"`` (popularity) or ``"user"`` (activity).
    relative_to
        ``"interactions"``: the share of all the interactions that involve the item,
        :math:`n_i / N`, which sums to 1 over the items (Vargas & Castells, 2011).
        ``"population"``: the fraction of users who interacted with the item,
        :math:`|U_i| / |U|`, where repeated interactions count once (Zhou et al.,
        2010; Abdollahpouri et al., 2019). With ``by="user"`` the two are the share of
        the interactions made by the user and the fraction of items they interacted
        with.

    Returns
    -------
    pandas.DataFrame
        Columns ``[by, "popularity"]``, one row per item (user), sorted by identifier.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table is invalid in one of the ways listed in :ref:`input-errors`; or if
        ``by`` or ``relative_to`` is not one of its options.

    References
    ----------
    * S. Vargas and P. Castells. *Rank and relevance in novelty and diversity metrics
      for recommender systems*. RecSys 2011. https://doi.org/10.1145/2043932.2043955
    * T. Zhou et al. *Solving the apparent diversity-accuracy dilemma of recommender
      systems*. PNAS 107(10), 2010. https://doi.org/10.1073/pnas.1000488107
    * H. Abdollahpouri, M. Mansoury, R. Burke and B. Mobasher. *The unfairness of
      popularity bias in recommendation*. RMSE @ RecSys 2019.
      https://arxiv.org/abs/1907.13286
    """
    by = _check_side(interactions, by)
    check_choice("relative_to", relative_to, ("interactions", "population"))
    codes, entities = encode_sorted(interactions, by)
    if relative_to == "interactions":
        counts = _core.interaction_counts(codes, len(entities))
        total = len(interactions)
    else:
        other = USER if by == ITEM else ITEM
        other_codes, others = encode_sorted(interactions, other)
        counts = _core.distinct_counts(codes, other_codes, len(entities))
        total = len(others)
    # Without interactions there is nothing to share out: the table is empty.
    shares = _core.shares(counts, total) if total else np.empty(0, dtype=np.float64)
    return pd.DataFrame({by: entities, _POPULARITY: shares})


def by_feature(features: pd.DataFrame, column: str, *, missing: Any = "unknown") -> pd.DataFrame:
    """Segment users (or items) by a categorical attribute, e.g. gender or genre.

    The groups are the values of the attribute. Users (items) whose value is missing
    are not dropped: they form a group of their own, as in Ekstrand et al. (2018).

    Parameters
    ----------
    features
        One row per user (item): a ``user`` or an ``item`` column, and the attribute.
    column
        Name of the column holding the attribute. It can itself be ``user`` or
        ``item`` in a table that has both: the other one is then what is grouped,
        e.g. ``by_feature(pairs, "item")`` groups the users by their item.
    missing
        Label of the group of the users (items) without a value: a scalar (a string,
        a number, ...) that is not itself a missing value (``None``, ``NaN``, ...).

    Returns
    -------
    pandas.DataFrame
        Columns ``["user" | "item", "group"]``, in the order of ``features``.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table is invalid in one of the ways listed in :ref:`input-errors`; if
        ``features`` has both a ``user`` and an ``item`` column next to the attribute, or
        neither; if ``column`` is not the name of a column; if a user
        (item) is listed more than once; if ``missing`` is not a scalar or is itself a
        missing value; or if it is already one of the values of the attribute while some
        values are missing.

    References
    ----------
    * M. D. Ekstrand et al. *All the cool kids, how do they fit in?: cold-start
      recommendations, online user-level, and demographic effects on recommender
      evaluation*. FAT* 2018. https://proceedings.mlr.press/v81/ekstrand18b.html
    * Y. Deldjoo et al. *A flexible framework for evaluating user and item fairness
      in recommender systems*. UMUAI 31, 2021.
      https://doi.org/10.1007/s11257-020-09285-1
    """
    if not pd.api.types.is_hashable(column):
        raise InvalidInputError("column must be the name of a column of features")
    check_frame(features, "features", [column])
    # `column` can itself be `user` or `item`: the other one is then what is grouped.
    sides = [side for side in (USER, ITEM) if side in features.columns and side != column]
    if len(sides) != 1:
        raise InvalidInputError(
            f"features must have either a {USER!r} or an {ITEM!r} column, "
            f"got the columns {list(features.columns)}"
        )
    side = sides[0]
    check_frame(features, "features", [column, side])
    if not pd.api.types.is_scalar(missing):
        raise InvalidInputError("missing must be a scalar label (a string, a number, ...)")
    if pd.isna(missing):
        raise InvalidInputError("missing must be a label, not a missing value")
    if features[side].isna().any():
        raise InvalidInputError(f"{side} identifiers must not be missing")
    if has_duplicates(features[side], side):
        raise InvalidInputError(f"features list the same {side} more than once")

    groups = features[column]
    absent = groups.isna()
    if absent.any():
        if (groups == missing).any():
            raise InvalidInputError(
                f"{missing!r} is already a value of {column!r}: "
                "choose another label for missing values"
            )
        groups = groups.astype(object).where(~absent, missing)
    return pd.DataFrame({side: features[side].to_numpy(), GROUP: groups.to_numpy()})
