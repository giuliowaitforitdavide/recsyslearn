"""Fairness metrics: how exposure or effectiveness is distributed over groups of users or items."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsyslearn.fairness._utils import eff_matrix, exp_matrix, prob_matrix
from recsyslearn._utils import check_columns_exist, safe_divide

__all__ = ["Entropy", "FairnessMetric", "KullbackLeibler", "MutualInformation"]


class FairnessMetric(ABC):
    """
    Abstract base class of the fairness metrics.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Compute the metric on the given recommendation lists.
        """


class Entropy(FairnessMetric):
    """
    Entropy evaluator for recommender systems.
    """

    def evaluate(
        self, top_n: pd.DataFrame, rel_matrix: pd.DataFrame | None = None
    ) -> float:
        """
        Compute the entropy of a model by using its recommendation list.

        :param top_n: Top N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param rel_matrix: Relevant items for users. It could be, for example, the items with a rating >= threshold.
        :type rel_matrix: pd.DataFrame, default None
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group').
        :raises DivisionByZeroException: If the total exposure is zero, so that the distribution over groups is undefined.
        :return: The computed entropy.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", "group"])

        top_n = (
            eff_matrix(top_n, rel_matrix, metric="Entropy")
            if rel_matrix is not None
            else top_n
        )
        top_n = prob_matrix(top_n, metric="Entropy")
        top_n = top_n[["group", "rank"]].groupby("group", as_index=False).sum()
        top_n["rank"] = top_n["rank"] * np.log2(top_n["rank"])
        return -top_n["rank"].sum()


class KullbackLeibler(FairnessMetric):
    """
    Kullback-Leibler divergence evaluator for recommender systems.
    """

    def evaluate(
        self,
        top_n: pd.DataFrame,
        target_representation: pd.DataFrame,
        rel_matrix: pd.DataFrame | None = None,
    ) -> float:
        """
        Compute the Kullback-Leibler divergence of a model, for a given target representation, by using its recommendation list.

        Groups with zero probability contribute 0 to the divergence (convention 0 · log(0 / q) = 0).

        :param top_n: Top N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param target_representation: The target representation desired for each group.
        :type target_representation: pd.DataFrame
        :param rel_matrix: Relevant items for users. It could be, for example, the items with a rating >= threshold.
        :type rel_matrix: pd.DataFrame, default None
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group') or if target_representation not in the form ('group', 'target_representation').
        :raises DivisionByZeroException: If the total exposure is zero, if a rank is 0, or if a group with non-zero probability has a target representation of 0 (infinite divergence).
        :return: The computed KL Divergence for the given target representation.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", "group"])
        check_columns_exist(target_representation, ["group", "target_representation"])

        top_n = (
            eff_matrix(top_n, rel_matrix, metric="KullbackLeibler")
            if rel_matrix is not None
            else exp_matrix(top_n, metric="KullbackLeibler")
        )
        top_n = prob_matrix(top_n, metric="KullbackLeibler")
        top_n = top_n[["group", "rank"]].groupby("group", as_index=False).sum()
        top_n = top_n.merge(target_representation, on="group")
        support = top_n[top_n["rank"] > 0]
        ratio = safe_divide(
            support["rank"],
            support["target_representation"],
            metric="KullbackLeibler",
            labels=support["group"],
        )
        return (support["rank"] * np.log2(ratio)).sum()


class MutualInformation(FairnessMetric):
    """
    Mutual Information evaluator for recommender systems.
    """

    def evaluate(
        self,
        top_n: pd.DataFrame,
        flag: str,
        rel_matrix: pd.DataFrame | None = None,
    ) -> float:
        """
        Compute the Mutual Information of a model by using its recommendation list.

        Pairs with zero joint probability contribute 0 to the sum (convention 0 · log(0 / q) = 0).

        :param top_n: Top N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param flag: Which actor of the recommendation scenario has been segmented (i.e. user).
        :type flag: str
        :param rel_matrix: Relevant items for users. It could be, for example, the items with a rating >= threshold.
        :type rel_matrix: pd.DataFrame, default None
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group').
        :raises KeyError: If flag is neither 'user' nor 'item'.
        :raises DivisionByZeroException: If the total exposure is zero or if a rank is 0.
        :return: The computed Mutual Information.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", "group"])

        not_flagged = {"user": "item", "item": "user"}
        if flag not in not_flagged:
            raise KeyError(f"Invalid flag {flag!r}, expected 'user' or 'item'")

        top_n = (
            eff_matrix(top_n, rel_matrix, metric="MutualInformation")
            if rel_matrix is not None
            else exp_matrix(top_n, metric="MutualInformation")
        )
        top_n = prob_matrix(top_n, metric="MutualInformation")
        not_grouped = not_flagged.get(flag)
        P_xy = (
            top_n[[not_grouped, "group", "rank"]]
            .groupby([not_grouped, "group"], as_index=False)
            .sum()
        )
        P_xP_y = (
            top_n[[not_grouped, "group", "rank"]]
            .groupby(not_grouped, as_index=False)
            .agg({"rank": "sum"})
        )
        P_xP_y = P_xy[[not_grouped, "group"]].merge(P_xP_y, on=not_grouped)
        tmp = top_n[["group", "rank"]].groupby("group", as_index=False).sum()
        P_xP_y = P_xP_y.merge(tmp, on=["group"])
        P_xP_y["rank"] = P_xP_y["rank_x"] * P_xP_y["rank_y"]
        tmp = P_xP_y[[not_grouped, "group", "rank"]].merge(
            P_xy, on=[not_grouped, "group"]
        )
        support = tmp[tmp["rank_y"] > 0]
        ratio = safe_divide(
            support["rank_y"],
            support["rank_x"],
            metric="MutualInformation",
            labels=support["group"],
        )
        return (support["rank_y"] * np.log2(ratio)).sum()
