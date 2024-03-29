from abc import ABC

import numpy as np
import pandas as pd

from recsyslearn.fairness.utils import eff_matrix, exp_matrix, prob_matrix
from recsyslearn.utils import check_columns_exist


class FairnessMetric(ABC):

    """
    Abstract Class for Metrics.
    """

    def __init__(self) -> None:
        return


class Entropy(FairnessMetric):

    """
    Entropy evaluator for recommender systems.
    """

    def evaluate(self, top_n: pd.DataFrame, rel_matrix: pd.DataFrame = None) -> float:
        """
        Compute the entropy of a model by using its recommendation list.

        :param top_n: Top N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param rel_matrix: Relevant items for users. It could be, for example, the items with a rating >= threshold.
        :type rel_matrix: pd.DataFrame, default None
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group').
        :return: The computed entropy.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", "group"])

        top_n = eff_matrix(top_n, rel_matrix) if rel_matrix is not None else top_n
        top_n = prob_matrix(top_n)
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
        rel_matrix: pd.DataFrame = None,
    ) -> float:
        """
        Compute the Kullback-Leibler divergence of a model, for a given target representation, by using its recommendation list.

        :param top_n: Top N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param target_representation: The target representation desired for each group.
        :type target_representation: pd.DataFrame
        :param rel_matrix: Relevant items for users. It could be, for example, the items with a rating >= threshold.
        :type rel_matrix: pd.DataFrame, default None
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group') or if target_representation not in the form ('group', 'target_representation').
        :return: The computed KL Divergence for the given target representation.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", "group"])
        check_columns_exist(target_representation, ["group", "target_representation"])

        top_n = (
            eff_matrix(top_n, rel_matrix)
            if rel_matrix is not None
            else exp_matrix(top_n)
        )
        top_n = prob_matrix(top_n)
        top_n = top_n[["group", "rank"]].groupby("group", as_index=False).sum()
        top_n = top_n.merge(target_representation, on="group")
        top_n["rank"] = top_n["rank"] * np.log2(
            top_n["rank"] / top_n["target_representation"]
        )
        return top_n["rank"].sum()


class MutualInformation(FairnessMetric):

    """
    Mutual Information evaluator for recommender systems.
    """

    def evaluate(
        self, top_n: pd.DataFrame, flag: str, rel_matrix: pd.DataFrame = None
    ) -> float:
        """
        Compute the Mutual Information of a model by using its recommendation list.

        :param top_n: Top N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param flag: Which actor of the recommendation scenario has been segmented (i.e. user).
        :type flag: str
        :param rel_matrix: Relevant items for users. It could be, for example, the items with a rating >= threshold.
        :type rel_matrix: pd.DataFrame, default None
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group').
        :return: The computed Mutual Information.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", "group"])

        not_flagged = {"user": "item", "item": "user"}

        top_n = (
            eff_matrix(top_n, rel_matrix)
            if rel_matrix is not None
            else exp_matrix(top_n)
        )
        top_n = prob_matrix(top_n)
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
        tmp["rank"] = tmp["rank_y"] * np.log2(tmp["rank_y"] / tmp["rank_x"])
        return tmp["rank"].sum()
