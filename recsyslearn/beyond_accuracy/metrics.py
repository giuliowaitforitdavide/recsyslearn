"""Beyond-accuracy metrics: properties of the recommendations other than their accuracy."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsyslearn._utils import describe_labels, check_columns_exist, safe_divide
from recsyslearn.errors.validation import InvalidPopularityException
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

    The novelty of an item is -log2(p(i)), where the popularity p(i) is the probability that an
    interaction involves item i, i.e. its share of interactions, as computed by
    PopularityPercentage. Less popular items are more novel.

    Novelty is a per-user metric: evaluate returns the novelty of each user's list and their mean.
    """

    def evaluate(
        self, top_n: pd.DataFrame, popularity_column: str = "percentage"
    ) -> MetricResult:
        """
        Compute the novelty of each user's recommendation list, and the mean over users.

        :param top_n: Top-N recommendations' lists for every user, with the popularity of each item.
        :type top_n: pd.DataFrame
        :param popularity_column: The column of top_n with the popularity of each item, a probability in (0, 1] such as the share of interactions from PopularityPercentage.
        :type popularity_column: str, default 'percentage'
        :raises ColumnsNotExistException: If top_n not in the form ('user', 'item', 'rank', popularity_column).
        :raises InvalidPopularityException: If a popularity is not in (0, 1], or is missing.
        :return: The mean of -log2(popularity) over the list of each user in per_user, and the mean over users in aggregate.
        :rtype: MetricResult
        """

        check_columns_exist(top_n, ["user", "item", "rank", popularity_column])
        popularity = pd.to_numeric(top_n[popularity_column]).astype(np.float64)
        invalid = ~((popularity > 0) & (popularity <= 1))
        if invalid.any():
            raise InvalidPopularityException(
                "Novelty", describe_labels(top_n["item"], invalid.to_numpy())
            )
        per_user = (-np.log2(popularity)).groupby(top_n["user"]).mean()
        return MetricResult.from_per_user("Novelty", per_user)
