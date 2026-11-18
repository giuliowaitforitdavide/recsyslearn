from __future__ import annotations

import numpy as np
import pandas as pd

from recsyslearn.errors.metrics import DivisionByZeroException
from recsyslearn.errors.validation import ColumnsNotExistException

_MAX_LABELS_IN_MESSAGE = 5


def safe_divide(
    numerator: float | np.ndarray | pd.Series,
    denominator: float | np.ndarray | pd.Series,
    metric: str,
    labels: str | pd.Series | None = None,
) -> float | np.ndarray | pd.Series:
    """
    Divide numerator by denominator, raising DivisionByZeroException if any denominator is zero.

    This is the only place where metrics divide (see AGENTS.md, section 3): a zero denominator
    means that the metric is undefined for the given input, and it always raises the same exception.

    :param numerator: The dividend.
    :type numerator: float, np.ndarray or pd.Series
    :param denominator: The divisor, with the same shape as numerator or a scalar.
    :type denominator: float, np.ndarray or pd.Series
    :param metric: Name of the metric, shown in the error message.
    :type metric: str
    :param labels: What the denominator refers to, shown in the error message: a string for a scalar
        denominator (e.g. "user 5"), or a Series aligned with the denominator whose name and values
        identify the zero entries (e.g. the group column).
    :type labels: str, pd.Series or None, default None
    :raises DivisionByZeroException: If any denominator is zero.
    :return: numerator / denominator.
    :rtype: float, np.ndarray or pd.Series
    """

    is_zero = np.asarray(denominator) == 0
    if is_zero.any():
        raise DivisionByZeroException(metric, _describe_zero_labels(labels, is_zero))
    return numerator / denominator


def _describe_zero_labels(
    labels: str | pd.Series | None, is_zero: np.ndarray
) -> str | None:
    """
    Describe which entries have a zero denominator, for the message of DivisionByZeroException.

    The labels are sorted, so the message does not depend on the order of the input rows.

    :param labels: The labels passed to safe_divide.
    :type labels: str, pd.Series or None
    :param is_zero: Boolean mask of the zero denominators.
    :type is_zero: np.ndarray
    :return: A description such as "group b, c", or None when no labels are given.
    :rtype: str or None
    """

    if labels is None or isinstance(labels, str):
        return labels
    if is_zero.ndim == 0:
        return labels.name
    zero_labels = _sorted(pd.unique(np.asarray(labels)[is_zero.ravel()]))
    shown = ", ".join(str(label) for label in zero_labels[:_MAX_LABELS_IN_MESSAGE])
    if len(zero_labels) > _MAX_LABELS_IN_MESSAGE:
        shown += f" and {len(zero_labels) - _MAX_LABELS_IN_MESSAGE} more"
    return f"{labels.name} {shown}" if labels.name else shown


def _sorted(labels: np.ndarray) -> list:
    """
    Sort labels by value, or by their text when they cannot be compared (e.g. mixed types).

    :param labels: The labels to sort.
    :type labels: np.ndarray
    :return: The sorted labels.
    :rtype: list
    """

    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=str)


def check_columns_exist(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Raise ColumnsNotExistException if pd.Dataframe does not contain the expected columns.

    :param df: Input that should be tested.
    :type df: pd.DataFrame
    :param columns: pd.DataFrame columns that should be contained.
    :type columns: list[str]
    :raises ColumnsNotExistException: If input does not contained expected columns.
    :return: The DataFrame with the expected columns and types.
    :rtype: pd.DataFrame
    """

    if not set(columns).issubset(set(df.columns)):
        raise ColumnsNotExistException(columns)

    dtypes = {
        "user": str,
        "item": str,
        "rank": float,
        "group": str,
        "target_representation": float,
    }

    return df.astype({col: dtypes[col] for col in df.columns if col in dtypes.keys()})
