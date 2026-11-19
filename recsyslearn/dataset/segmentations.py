"""Segmentations of users and items into groups."""

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import pandas as pd

from recsyslearn.errors.segmentation import (
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)
from recsyslearn.errors.validation import InvalidGroupException

__all__ = [
    "ActivitySegmentation",
    "DiscreteFeatureSegmentation",
    "InteractionSegmentation",
    "PopularityPercentage",
    "Segmentation",
]


class Segmentation(ABC):
    """
    Abstract base class of the segmentations.
    """

    @abstractmethod
    def segment(self, *args, **kwargs):
        """
        Assign the users or items of the given data to groups.
        """


class InteractionSegmentation(Segmentation):
    """
    Segmentation of items based on the number of cumulative interaction they have.
    """

    def segment(
        self,
        dataset: pd.DataFrame,
        proportions: list[float] | None = None,
        min_interaction: int = 0,
        group: str = "item",
    ) -> pd.DataFrame:
        """
        Segmentation of items based on their cumulative interactions with different users.

        Items are ranked from the most to the least interacted, ties broken by id. An item joins
        group g when the items ranked before it account for at least the proportions of the groups
        before g, and for less than the proportions up to g. So each group is the smallest set of
        the most interacted remaining items that covers its proportion, including the item that
        crosses the threshold, and the most interacted item is always in the first group. A group
        is empty when the earlier groups already cover its proportion (e.g. an item alone above
        the first two proportions). Items with at most min_interaction interactions go to the
        last group.

        :param dataset: The complete dataset.
        :type dataset: pd.DataFrame
        :param proportions: The proportion of interactions wanted for every group.
        :type proportions: list[float] or None, default [0.8, 0.2]
        :param min_interaction: Items with at most this number of interactions are not counted and go to the last group.
        :type min_interaction: int, default 0
        :param group: The group which has to be segmented based on their number of interaction.
        :type group: str, default 'item'
        :raises SegmentationNotSupportedException: If len(proportions) not in (1, 2, 3).
        :raises WrongProportionsException: If sum(proportions) is not 1, which means it doesn't cover all the items/users.
        :raises InvalidGroupException: If group is not equal to 'user' or 'item'.
        :return: DataFrame with items and belonging group.
        :rtype: pd.DataFrame
        """

        if proportions is None:
            proportions = [0.8, 0.2]

        if len(proportions) == 1:
            return dataset

        if len(proportions) not in (2, 3):
            raise SegmentationNotSupportedException(
                "Number of supported group is between 1 and 3."
            )

        if np.sum(proportions * 10) / 10 != 1:
            raise WrongProportionsException()

        if group not in ["user", "item"]:
            raise InvalidGroupException(group)

        # groupby returns the ids sorted, so a stable sort breaks ties by id
        item_groups = (
            dataset.groupby(group)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False, kind="stable")
        )

        counted = item_groups["count"].where(item_groups["count"] > min_interaction, 0)
        n_int = counted.sum()
        before = counted.cumsum() - counted
        # Small tolerance, so that 0.6 + 0.3 (= 0.8999…) still reaches 90% of the interactions
        thresholds = np.cumsum(proportions[:-1]) * n_int - 1e-9 * max(n_int, 1)
        groups = np.searchsorted(thresholds, before.to_numpy(), side="right") + 1
        item_groups["group"] = np.where(counted > 0, groups, len(proportions))
        return item_groups[[group, "group"]].astype({f"{group}": str, "group": str})


class PopularityPercentage(Segmentation):
    """
    Calculate item or user popularity based on the percentage of interaction they have.
    """

    def segment(self, dataset: pd.DataFrame, group: str = "item") -> pd.DataFrame:
        """
        Calculate item or user popularity based on the percentage of interaction they have.

        :param dataset: The complete dataset.
        :type dataset: pd.DataFrame
        :param group: Whether to calculate the popularity of users or items.
        :type group: str, default 'item'
        :return: DataFrame with items/user and corresponding popularity.
        :rtype: pd.DataFrame
        """

        item_interactions = dataset[group].values
        total_interactions = len(item_interactions)
        inter_counter = Counter(item_interactions)
        inter_counter = {
            item: counts / total_interactions for item, counts in inter_counter.items()
        }

        popularity_dataframe = pd.DataFrame.from_dict(
            inter_counter, orient="index"
        ).reset_index()
        popularity_dataframe.columns = [group, "percentage"]

        return popularity_dataframe


class ActivitySegmentation(Segmentation):
    """
    Segmentation of users based on their number of interaction.
    """

    def segment(
        self,
        dataset: pd.DataFrame,
        proportions: list[float] | None = None,
        min_interaction: int = 0,
    ) -> pd.DataFrame:
        """
        Segmentation of users based on their interactions with different items.

        Users are ranked by their number of interactions, from the most to the least active, and
        split into groups by position in that ranking. Users with the same number of interactions
        are ordered by user id, so the same dataset always gives the same groups, whatever the
        order of its rows.

        :param dataset: The complete dataset.
        :type dataset: pd.DataFrame
        :param proportions: The share of users wanted for every group, from the most active one.
        :type proportions: list[float] or None, default [0.1, 0.9]
        :param min_interaction: The minimum number of interaction allowed per user. Users below this threshold will be removed.
        :type min_interaction: int, default 0
        :raises SegmentationNotSupportedException: If len(proportion) not in (1, 2, 3).
        :raises WrongProportionsException: If sum(proportion) is not 1, which means it doesn't cover all the items/users.
        :return: DataFrame with users and belonging group.
        :rtype: pd.DataFrame
        """

        if proportions is None:
            proportions = [0.1, 0.9]

        if len(proportions) == 1:
            return dataset

        if len(proportions) not in (2, 3):
            raise SegmentationNotSupportedException(
                "Number of supported group is between 1 and 3."
            )

        if np.sum(proportions * 10) / 10 != 1:
            raise WrongProportionsException()

        user_groups = dataset.groupby("user").size().reset_index(name="count")
        user_groups = user_groups.loc[user_groups["count"] >= min_interaction, :]

        # groupby returns the users sorted by id, so a stable sort breaks ties by user id
        user_groups = user_groups.sort_values("count", ascending=False, kind="stable")
        user_groups.loc[:, "count"] = np.arange(user_groups.shape[0]) + 1
        first_thr = np.rint(proportions[0] * user_groups.shape[0])
        second_thr = np.rint(proportions[1] * user_groups.shape[0]) + first_thr
        first_thr = first_thr if first_thr > 0 else 1
        first_group = user_groups.loc[user_groups["count"] <= first_thr, "user"]
        second_group = user_groups.loc[user_groups["count"].lt(second_thr), "user"]

        conditions = [
            user_groups["user"].isin(first_group),
            user_groups["user"].isin(second_group),
        ]
        choices = (1, 2)
        default = len(proportions)
        user_groups.loc[:, "group"] = np.select(conditions, choices, default=default)

        return user_groups[["user", "group"]].astype({"user": str, "group": str})


class DiscreteFeatureSegmentation(Segmentation):
    """
    Segmentation of entities (users or items) according to one of
    their features (e.g., gender for users or genre for items)
    """

    def segment(self, feature: pd.DataFrame, fill_na: int = -1) -> pd.DataFrame:
        """
        Segmentation of users/items based on one of their features.
        Before assigning the group, the nans are given a -1 value by default.
        Make sure that this is not one of the feature values, already.

        :param feature: The feature dataframe in form of [id, feature] storing the categorical feature to be used for grouping.
        :type feature: pd.DataFrame
        :param fill_na: The value with which to fill not assigned values. Default is -1.
        :type fill_na: int, default -1
        :raises InvalidValueException: If the fill_na value is already present in the features dataframe.
        :return: DataFrame with items and belonging group.
        :rtype: pd.DataFrame
        """

        column = feature.columns[1]
        values = feature[column]

        if fill_na in values.unique():
            raise InvalidValueException(fill_na)

        # The str dtype (default for strings since pandas 3) rejects a non-string fill_na
        if isinstance(values.dtype, pd.StringDtype):
            values = values.astype(object)
        groups = values.fillna(fill_na).astype("category").cat.codes

        # A shallow copy shares the data; assigning the column replaces it only in the copy
        segmented = feature.copy(deep=False)
        segmented[column] = groups
        return segmented.rename(columns={column: "group"})
