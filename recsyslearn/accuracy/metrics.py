"""Accuracy metrics: how well the recommended items match the relevant ones."""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsyslearn.errors.metrics import RecListTooShortException
from recsyslearn.utils import check_columns_exist, safe_divide

__all__ = ["AccuracyMetric", "NDCG"]


class AccuracyMetric(ABC):
    """
    Abstract base class of the accuracy metrics.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Compute the metric on the given recommendation lists.
        """


class NDCG(AccuracyMetric):
    """
    NDCG evaluator for recommender systems.
    """

    @staticmethod
    def _dcg(scores: np.ndarray) -> float:
        """DCG of the given relevance scores, in ranking order, with gain 2^rel - 1."""
        return np.sum(
            safe_divide(
                np.power(2, scores) - 1,
                np.log(np.arange(scores.shape[0], dtype=np.float32) + 2),
                metric="NDCG",
            ),
            dtype=np.float32,
        )

    def _ndcg(
        self,
        ranked_list: np.ndarray,
        pos_items: np.ndarray,
        user: str,
        at: int,
    ) -> float:
        """NDCG@at of one user, with binary relevance of the items in pos_items."""
        relevance = np.ones_like(pos_items, dtype=np.int32)
        it2rel = {it: r for it, r in zip(pos_items, relevance)}

        rank_scores = np.asarray(
            [it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32
        )
        ideal_dcg = self._dcg(np.sort(relevance)[::-1][:at])
        rank_dcg = self._dcg(rank_scores)

        return float(
            safe_divide(rank_dcg, ideal_dcg, metric="NDCG", labels=f"user {user}")
        )

    def evaluate(
        self,
        top_n: pd.DataFrame,
        pos_items: pd.DataFrame,
        ats: tuple[int, ...] = (5, 10),
    ) -> pd.DataFrame:
        """Compute the NDCG@k of each user from their recommendation list.

        :param top_n: Top N recommendations' lists for every user. Columns: ['user', 'item', 'rank'].
        :type top_n: pd.DataFrame
        :param pos_items: Relevant items per user. Columns: ['user', 'pos_items'].
        :type pos_items: pd.DataFrame
        :param ats: The tuple of values at which to evaluate NDCG@k.
        :type ats: tuple, default (5, 10)
        :raises ColumnsNotExistException: If top_n does not contain columns ('user', 'item', 'rank') or pos_items does not contain columns ('user', 'pos_items').
        :raises RecListTooShortException: If the top_n list does not contain enough items.
        :raises DivisionByZeroException: If a user has no relevant items, so that the ideal DCG is zero and NDCG is undefined. Filter such users out of pos_items before calling evaluate.
        :return: The NDCG@k of each user, in the form ('user', 'NDCG@k_0', ..., 'NDCG@k_n'), with one column for each computable cut-off.
        :rtype: pd.DataFrame
        """

        check_columns_exist(top_n, ["user", "item", "rank"])
        check_columns_exist(pos_items, ["user", "pos_items"])

        min_calculable_at = top_n.groupby("user").size().min()
        calculable_ats = [k for k in ats if k <= min_calculable_at]

        if len(calculable_ats) == 0:
            raise RecListTooShortException(ats)

        non_calculable_ats = [k for k in ats if k > min_calculable_at]
        if non_calculable_ats:
            warnings.warn(
                f"NDCG@k won't be calculated for k in {non_calculable_ats}: "
                f"the shortest recommendation list has {min_calculable_at} items."
            )

        top_n = top_n[["user", "item", "rank"]]
        top_n = (
            top_n[["user", "item"]]
            .groupby("user")["item"]
            .apply(np.asarray)
            .reset_index()
        )
        full_df = top_n.merge(pos_items, on="user")

        for k in calculable_ats:
            full_df.loc[:, f"NDCG@{k}"] = full_df.apply(
                lambda x: self._ndcg(x["item"][:k], x["pos_items"], x["user"], at=k),
                axis=1,
            )

        cols_to_be_returned = [
            col for col in full_df.columns if col not in ["item", "pos_items"]
        ]

        return full_df.loc[:, cols_to_be_returned]
