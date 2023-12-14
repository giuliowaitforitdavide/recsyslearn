from abc import ABC
from collections import Counter

import numpy as np
import pandas as pd

from recsyslearn.errors.errors import (
    InvalidGroupException,
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)


class Segmentation(ABC):

    """
    Abstract class for different group segmentations.
    """

    def __init__(self) -> None:
        super().__init__()


class InteractionSegmentation(Segmentation):

    """
    Segmentation of items based on the number of cumulative interaction they have.
    """

    @classmethod
    def segment(
        cls,
        dataset: pd.DataFrame,
        proportions=None,
        min_interaction: int = 0,
        group="item",
    ) -> pd.DataFrame:
        """
        Segmentation of items based on their cumulative interactions with different users.

        :param dataset: The complete dataset.
        :type dataset: pd.DataFrame
        :param proportions: The proportion of interactions wanted for every group.
        :type proportions: list, default [0.8, 0.2]
        :param min_interaction: The minimum number of interaction allowed for items. Items below this threshold will be removed.
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

        item_groups = (
            dataset.groupby(group)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        tmp = item_groups[item_groups["count"] > min_interaction]
        n_int = tmp["count"].sum()

        short_thr = np.rint(proportions[0] * n_int)
        mid_thr = np.rint(proportions[1] * n_int) + short_thr
        tmp.loc[:, "cumulative_sum"] = tmp["count"].cumsum()

        short_head = tmp.loc[tmp["cumulative_sum"].lt(short_thr), group]
        mid_tail = tmp.loc[
            tmp["cumulative_sum"].lt(mid_thr) & ~tmp[group].isin(short_head), group
        ]
        conditions = [
            item_groups[group].isin(short_head),
            item_groups[group].isin(mid_tail),
        ]
        choices = (1, 2)
        default = len(proportions)
        item_groups.loc[:, "group"] = np.select(conditions, choices, default=default)
        return item_groups[[group, "group"]].astype({f"{group}": str, "group": str})


class PopularityPercentage(Segmentation):

    """
    Calculate item or user popularity based on the percentage of interaction they have.
    """

    @classmethod
    def segment(cls, dataset: pd.DataFrame, group: str = "item") -> pd.DataFrame:
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

    @classmethod
    def segment(
        cls, dataset: pd.DataFrame, proportions=None, min_interaction: int = 0
    ) -> pd.DataFrame:
        """
        Segmentation of users based on their interactions with different items.

        :param dataset: The complete dataset.
        :type dataset: pd.DataFrame
        :param proportions: The proportion of interactions wanted for every group.
        :type proportions: list, default [0.8, 0.2]
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

        user_groups.loc[:, "count"] = user_groups.loc[:, "count"].apply(
            lambda x: x + np.random.choice(list(range(10)))
        )
        user_groups = user_groups.sort_values("count", ascending=False)
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

    @classmethod
    def segment(cls, feature: pd.DataFrame, fill_na: int = -1) -> pd.DataFrame:
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

        if fill_na in feature[feature.columns[1]].unique():
            raise InvalidValueException(fill_na)

        feature = feature.fillna({str(feature.columns[1]): fill_na})

        feature.loc[:, feature.columns[1]] = (
            feature[feature.columns[1]].astype("category").cat.codes
        )
        feature = feature.rename({str(feature.columns[1]): "group"}, axis="columns")
        return feature
