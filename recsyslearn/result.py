"""The result type returned by every metric."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["MetricResult"]


@dataclass(frozen=True, eq=False)
class MetricResult:
    """
    Value of a metric on a set of recommendation lists.

    Every metric returns a MetricResult, so calling code does not need to tell metrics apart.
    Results are compared by identity: compare their fields to check that two results are equal.

    :param name: Name of the metric, with its cut-off where it has one (e.g. "NDCG@10", "Coverage").
    :type name: str
    :param aggregate: For per-user metrics, the mean of per_user over users (macro-average: each
        user weighs equally); for population-level metrics, the value over the whole population.
    :type aggregate: float
    :param per_user: The value of each user, indexed by user and named after the metric; None for
        population-level metrics, which are not defined per user.
    :type per_user: pd.Series or None, default None
    """

    name: str
    aggregate: float
    per_user: pd.Series | None = None

    @classmethod
    def from_per_user(cls, name: str, per_user: pd.Series) -> MetricResult:
        """
        Build the result of a per-user metric, with the macro-average over users as aggregate.

        :param name: Name of the metric.
        :type name: str
        :param per_user: The value of each user, indexed by user.
        :type per_user: pd.Series
        :return: The result, with per_user renamed after the metric and its index named "user".
        :rtype: MetricResult
        """

        per_user = per_user.rename(name).rename_axis("user")
        return cls(name=name, aggregate=float(per_user.mean()), per_user=per_user)
