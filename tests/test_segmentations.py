import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from recsyslearn.dataset.segmentations import (
    ActivitySegmentation,
    DiscreteFeatureSegmentation,
    InteractionSegmentation,
    PopularityPercentage,
)
from recsyslearn.errors.segmentation import (
    InvalidGroupException,
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)
from tests.fixtures import (
    item_genres,
    item_interactions,
    popularity_interactions,
    user_ages,
    user_genders,
    user_interactions,
)


def groups_of(segmented: pd.DataFrame, key: str, entity: str) -> set:
    """The groups assigned to one user or item in a segmentation result."""
    return set(segmented.loc[segmented[key] == entity, "group"])


def percentages(popularity: pd.DataFrame, key: str) -> dict:
    """Map each user or item to its popularity percentage."""
    return dict(zip(popularity[key], popularity["percentage"]))


class InteractionSegmentationTest(unittest.TestCase):
    def test_most_interacted_item_goes_to_the_first_group(self) -> None:
        segmented = InteractionSegmentation().segment(item_interactions())
        self.assertEqual(groups_of(segmented, "item", "1"), {"1"})

    def test_least_interacted_item_goes_to_the_last_group(self) -> None:
        segmented = InteractionSegmentation().segment(item_interactions())
        self.assertEqual(groups_of(segmented, "item", "2"), {"2"})

    def test_accepts_proportions_with_floating_point_rounding(self) -> None:
        """0.6 + 0.3 + 0.1 is not exactly 1 in floating point, but must be accepted."""
        InteractionSegmentation().segment(item_interactions(), [0.6, 0.3, 0.1])

    def test_a_single_group_returns_the_dataset_unchanged(self) -> None:
        assert_frame_equal(
            InteractionSegmentation().segment(item_interactions(), [1]),
            item_interactions(),
        )

    def test_raises_when_there_are_more_groups_than_items(self) -> None:
        with self.assertRaises(SegmentationNotSupportedException):
            InteractionSegmentation().segment(item_interactions(), [0.7, 0.1, 0.1, 0.1])

    def test_raises_when_proportions_do_not_sum_to_one(self) -> None:
        with self.assertRaises(WrongProportionsException):
            InteractionSegmentation().segment(item_interactions(), [0.7, 0.4])

    def test_raises_for_an_invalid_group_name(self) -> None:
        with self.assertRaises(InvalidGroupException):
            InteractionSegmentation().segment(
                item_interactions(), [0.7, 0.3], 1, "users"
            )


class ActivitySegmentationTest(unittest.TestCase):
    def test_most_active_user_goes_to_the_first_group(self) -> None:
        segmented = ActivitySegmentation().segment(user_interactions())
        self.assertEqual(groups_of(segmented, "user", "1"), {"1"})

    def test_less_active_users_go_to_the_last_group(self) -> None:
        segmented = ActivitySegmentation().segment(user_interactions())
        for user in ("2", "3"):
            with self.subTest(user=user):
                self.assertEqual(groups_of(segmented, "user", user), {"2"})

    def test_a_single_group_returns_the_dataset_unchanged(self) -> None:
        assert_frame_equal(
            ActivitySegmentation().segment(user_interactions(), [1]),
            user_interactions(),
        )

    def test_raises_when_there_are_more_groups_than_users(self) -> None:
        with self.assertRaises(SegmentationNotSupportedException):
            ActivitySegmentation().segment(user_interactions(), [0.7, 0.1, 0.1, 0.1])

    def test_raises_when_proportions_do_not_sum_to_one(self) -> None:
        with self.assertRaises(WrongProportionsException):
            ActivitySegmentation().segment(user_interactions(), [0.7, 0.4])


class PopularityPercentageTest(unittest.TestCase):
    def test_item_percentage_is_its_share_of_interactions(self) -> None:
        popularity = PopularityPercentage().segment(popularity_interactions())
        self.assertEqual(
            percentages(popularity, "item"), {"1": 0.6, "2": 0.3, "3": 0.1}
        )

    def test_item_percentages_sum_to_one(self) -> None:
        popularity = PopularityPercentage().segment(popularity_interactions())
        self.assertAlmostEqual(popularity["percentage"].sum(), 1.0, delta=1e-5)

    def test_user_percentage_is_its_share_of_interactions(self) -> None:
        popularity = PopularityPercentage().segment(
            popularity_interactions(), group="user"
        )
        self.assertEqual(
            percentages(popularity, "user"), {"1": 0.7, "2": 0.2, "3": 0.1}
        )

    def test_user_percentages_sum_to_one(self) -> None:
        popularity = PopularityPercentage().segment(
            popularity_interactions(), group="user"
        )
        self.assertAlmostEqual(popularity["percentage"].sum(), 1.0, delta=1e-5)


class DiscreteFeatureSegmentationTest(unittest.TestCase):
    def test_users_with_different_values_get_different_groups(self) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=user_genders())
        self.assertNotEqual(
            groups_of(segmented, "user", "1"), groups_of(segmented, "user", "2")
        )

    def test_missing_values_get_their_own_group(self) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=user_genders())
        for user in ("1", "2"):
            with self.subTest(user=user):
                self.assertNotEqual(
                    groups_of(segmented, "user", user),
                    groups_of(segmented, "user", "3"),
                )

    def test_none_and_nan_are_the_same_missing_group(self) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=user_genders())
        self.assertEqual(
            groups_of(segmented, "user", "3"), groups_of(segmented, "user", "4")
        )

    def test_items_with_the_same_value_share_a_group(self) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=item_genres())
        self.assertEqual(
            groups_of(segmented, "item", "1"), groups_of(segmented, "item", "2")
        )

    def test_items_with_different_values_get_different_groups(self) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=item_genres())
        for first, second in (("1", "3"), ("2", "3"), ("3", "4")):
            with self.subTest(items=(first, second)):
                self.assertNotEqual(
                    groups_of(segmented, "item", first),
                    groups_of(segmented, "item", second),
                )

    def test_raises_when_the_feature_contains_the_fill_na_value(self) -> None:
        with self.assertRaises(InvalidValueException):
            DiscreteFeatureSegmentation().segment(feature=user_ages())


if __name__ == "__main__":
    unittest.main()
