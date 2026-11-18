"""Beyond-accuracy metrics: properties of the recommendations other than their accuracy."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsyslearn._utils import check_columns_exist, safe_divide

__all__ = ["BeyondAccuracyMetric", "Coverage", "Novelty"]


class BeyondAccuracyMetric(ABC):
    """
    Abstract base class of the beyond-accuracy metrics.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Compute the metric on the given recommendation lists.
        """


class Coverage(BeyondAccuracyMetric):
    """
    Coverage evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    """

    def evaluate(self, top_n: pd.DataFrame, items: list) -> float:
        """
        Compute the coverage of a model by using its recommendation list.

        :param top_n: Top-N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param items: List of items in the dataset.
        :type items: list or array-like
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group').
        :raises DivisionByZeroException: If items is empty, so that coverage is undefined.
        :return: The computed coverage.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank"])
        return safe_divide(
            len(top_n.item.unique().tolist()), len(items), metric="Coverage"
        )


class Novelty(BeyondAccuracyMetric):
    """
    Novelty evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    where popularity is defined in terms of the segmentation of the item groups
    (e.g. short head -> 3, mid tail   -> 2, long tail  -> 1)
    or in terms of percentage of user-item interactions.
    """

    def evaluate(
        self, top_n: pd.DataFrame, popularity_definition: str = "group"
    ) -> float:
        """
        Compute the novelty of a model by using its recommendation list and the segmented item groups.

        :param top_n: Top-N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param popularity_definition: Either 'group' or 'percentage', to choose whether popularity is computed in terms of
            segmenting items/users according to the distribution of user-item interactions
            or if it is defined as the percentage of user-item interactions.
        :type popularity_definition: str, default 'group'
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', popularity_definition).
        :return: The computed novelty.
        :rtype: float
        """

        check_columns_exist(top_n, ["user", "item", "rank", popularity_definition])
        top_n[popularity_definition] = pd.to_numeric(top_n[popularity_definition])
        top_n = top_n.groupby("user")[popularity_definition].apply(
            lambda x: np.mean(-np.log2(x.astype(float)))
        )
        return top_n.mean()
