"""Beyond-accuracy metrics: properties of the recommendations other than their accuracy."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsyslearn._utils import check_columns_exist, safe_divide
from recsyslearn.result import MetricResult

__all__ = ["BeyondAccuracyMetric", "Coverage", "Novelty"]


class BeyondAccuracyMetric(ABC):
    """
    Abstract base class of the beyond-accuracy metrics.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> MetricResult:
        """
        Compute the metric on the given recommendation lists.
        """


class Coverage(BeyondAccuracyMetric):
    """
    Coverage evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2

    Coverage is a population-level metric: it is defined over all the recommendation lists
    together, so it has no per-user values.
    """

    def evaluate(self, top_n: pd.DataFrame, items: list) -> MetricResult:
        """
        Compute the coverage of a model by using its recommendation list.

        :param top_n: Top-N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param items: List of items in the dataset.
        :type items: list or array-like
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', 'group').
        :raises DivisionByZeroException: If items is empty, so that coverage is undefined.
        :return: The coverage in aggregate; per_user is None.
        :rtype: MetricResult
        """

        check_columns_exist(top_n, ["user", "item", "rank"])
        coverage = safe_divide(top_n["item"].nunique(), len(items), metric="Coverage")
        return MetricResult(name="Coverage", aggregate=float(coverage))


class Novelty(BeyondAccuracyMetric):
    """
    Novelty evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    where popularity is defined in terms of the segmentation of the item groups
    (e.g. short head -> 3, mid tail   -> 2, long tail  -> 1)
    or in terms of percentage of user-item interactions.

    Novelty is a per-user metric: evaluate returns the novelty of each user's list and their mean.
    """

    def evaluate(
        self, top_n: pd.DataFrame, popularity_definition: str = "group"
    ) -> MetricResult:
        """
        Compute the novelty of a model by using its recommendation list and the segmented item groups.

        :param top_n: Top-N recommendations' lists for every user with items or users already segmented.
        :type top_n: pd.DataFrame
        :param popularity_definition: Either 'group' or 'percentage', to choose whether popularity is computed in terms of
            segmenting items/users according to the distribution of user-item interactions
            or if it is defined as the percentage of user-item interactions.
        :type popularity_definition: str, default 'group'
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', popularity_definition).
        :return: The mean of -log2(popularity) over the list of each user in per_user, and the mean over users in aggregate.
        :rtype: MetricResult
        """

        check_columns_exist(top_n, ["user", "item", "rank", popularity_definition])
        popularity = pd.to_numeric(top_n[popularity_definition]).astype(np.float64)
        per_user = (-np.log2(popularity)).groupby(top_n["user"]).mean()
        return MetricResult.from_per_user("Novelty", per_user)
