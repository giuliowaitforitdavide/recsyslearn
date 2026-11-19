"""Accuracy metrics: how well the recommended items match the relevant ones."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsyslearn._utils import check_columns_exist, safe_divide
from recsyslearn.errors.metrics import InvalidCutoffException, RecListTooShortException
from recsyslearn.result import MetricResult

__all__ = ["AccuracyMetric", "NDCG"]


class AccuracyMetric(ABC):
    """
    Abstract base class of the accuracy metrics.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> MetricResult:
        """
        Compute the metric on the given recommendation lists.
        """


class NDCG(AccuracyMetric):
    """
    NDCG@k evaluator for recommender systems.

    NDCG is a per-user metric: evaluate returns the NDCG@k of each user and their mean.
    It is computed in float64.

    :param k: The cut-off: only the first k items of each recommendation list are evaluated.
    :type k: int
    :raises InvalidCutoffException: If k is not a positive integer.
    """

    def __init__(self, k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise InvalidCutoffException("NDCG", k)
        self.k = int(k)

    @property
    def name(self) -> str:
        """The name of the metric with its cut-off, e.g. "NDCG@10"."""
        return f"NDCG@{self.k}"

    @staticmethod
    def _dcg(scores: np.ndarray) -> float:
        """DCG of the given relevance scores, in ranking order, with gain 2^rel - 1."""
        return np.sum(
            safe_divide(
                np.power(2, scores) - 1,
                np.log(np.arange(scores.shape[0], dtype=np.float64) + 2),
                metric="NDCG",
            ),
            dtype=np.float64,
        )

    def _ndcg(self, ranked_list: np.ndarray, pos_items: np.ndarray, user: str) -> float:
        """NDCG@k of one user, with binary relevance of the items in pos_items."""
        relevance = np.ones_like(pos_items, dtype=np.int64)
        it2rel = {it: r for it, r in zip(pos_items, relevance)}

        rank_scores = np.asarray(
            [it2rel.get(it, 0.0) for it in ranked_list[: self.k]], dtype=np.float64
        )
        ideal_dcg = self._dcg(np.sort(relevance)[::-1][: self.k])
        rank_dcg = self._dcg(rank_scores)

        return float(
            safe_divide(rank_dcg, ideal_dcg, metric="NDCG", labels=f"user {user}")
        )

    def evaluate(self, top_n: pd.DataFrame, pos_items: pd.DataFrame) -> MetricResult:
        """
        Compute the NDCG@k of each user from their recommendation list, and the mean over users.

        Each list is ordered by its rank column, so the row order of top_n does not matter. Users
        of top_n without an entry in pos_items are left out.

        :param top_n: Top N recommendations' lists for every user. Columns: ['user', 'item', 'rank'].
        :type top_n: pd.DataFrame
        :param pos_items: Relevant items per user. Columns: ['user', 'pos_items'].
        :type pos_items: pd.DataFrame
        :raises ColumnsNotExistException: If top_n does not contain columns ('user', 'item', 'rank') or pos_items does not contain columns ('user', 'pos_items').
        :raises RecListTooShortException: If k is longer than the shortest recommendation list.
        :raises DivisionByZeroException: If a user has no relevant items, so that the ideal DCG is zero and NDCG is undefined. Filter such users out of pos_items before calling evaluate.
        :return: The NDCG@k of each user in per_user, and their mean in aggregate.
        :rtype: MetricResult
        """

        check_columns_exist(top_n, ["user", "item", "rank"])
        check_columns_exist(pos_items, ["user", "pos_items"])

        shortest = int(top_n.groupby("user").size().min())
        if self.k > shortest:
            raise RecListTooShortException("NDCG", self.k, shortest)

        lists = (
            top_n[["user", "item", "rank"]]
            .sort_values(["user", "rank"], kind="stable")
            .groupby("user")["item"]
            .apply(np.asarray)
            .reset_index()
        )
        full_df = lists.merge(pos_items[["user", "pos_items"]], on="user")
        per_user = pd.Series(
            [
                self._ndcg(ranked, relevant, user)
                for user, ranked, relevant in zip(
                    full_df["user"], full_df["item"], full_df["pos_items"]
                )
            ],
            index=full_df["user"],
            dtype=np.float64,
        )
        return MetricResult.from_per_user(self.name, per_user)
