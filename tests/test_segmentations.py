import unittest

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from recsyslearn.dataset.segmentations import (
    ActivitySegmentation,
    DiscreteFeatureSegmentation,
    InteractionSegmentation,
    PopularityPercentage,
)
from recsyslearn.dataset.utils import find_relevant_items
from recsyslearn.errors.segmentation import (
    InvalidGroupException,
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)
from tests import strategies
from tests.case import TestCase
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


VALID_PROPORTIONS = st.sampled_from([[0.8, 0.2], [0.5, 0.5], [0.6, 0.3, 0.1]])


def percentages(popularity: pd.DataFrame, key: str) -> dict:
    """Map each user or item to its popularity percentage."""
    return dict(zip(popularity[key], popularity["percentage"]))


class InteractionSegmentationTest(TestCase):
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
        with self.assertRaisesMessage(
            SegmentationNotSupportedException,
            "Number of supported group is between 1 and 3.",
        ):
            InteractionSegmentation().segment(item_interactions(), [0.7, 0.1, 0.1, 0.1])

    def test_raises_when_proportions_do_not_sum_to_one(self) -> None:
        with self.assertRaisesMessage(
            WrongProportionsException, "Proportions doesn't cover all the items/users."
        ):
            InteractionSegmentation().segment(item_interactions(), [0.7, 0.4])

    def test_raises_for_an_invalid_group_name(self) -> None:
        with self.assertRaisesMessage(
            InvalidGroupException, "users is not a valid group"
        ):
            InteractionSegmentation().segment(
                item_interactions(), [0.7, 0.3], 1, "users"
            )

    @given(strategies.interactions(), VALID_PROPORTIONS)
    def test_assigns_exactly_one_group_to_every_item(self, data, proportions) -> None:
        segmented = InteractionSegmentation().segment(data, proportions)
        self.assertEqual(sorted(segmented["item"]), sorted(data["item"].unique()))

    @given(strategies.interactions(), VALID_PROPORTIONS)
    def test_uses_one_group_per_proportion_at_most(self, data, proportions) -> None:
        segmented = InteractionSegmentation().segment(data, proportions)
        allowed = {str(group) for group in range(1, len(proportions) + 1)}
        self.assertTrue(set(segmented["group"]) <= allowed)

    @given(strategies.interactions(), VALID_PROPORTIONS)
    def test_more_interacted_items_are_never_in_a_later_group(
        self, data, proportions
    ) -> None:
        segmented = InteractionSegmentation().segment(data, proportions)
        counts = data["item"].value_counts()
        group = dict(zip(segmented["item"], segmented["group"].astype(int)))
        for first in group:
            for second in group:
                if counts[first] > counts[second]:
                    with self.subTest(first=first, second=second):
                        self.assertLessEqual(group[first], group[second])


class ActivitySegmentationTest(TestCase):
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
        with self.assertRaisesMessage(
            SegmentationNotSupportedException,
            "Number of supported group is between 1 and 3.",
        ):
            ActivitySegmentation().segment(user_interactions(), [0.7, 0.1, 0.1, 0.1])

    def test_raises_when_proportions_do_not_sum_to_one(self) -> None:
        with self.assertRaisesMessage(
            WrongProportionsException, "Proportions doesn't cover all the items/users."
        ):
            ActivitySegmentation().segment(user_interactions(), [0.7, 0.4])


class PopularityPercentageTest(TestCase):
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

    @given(strategies.interactions())
    def test_generated_percentages_sum_to_one(self, data) -> None:
        popularity = PopularityPercentage().segment(data)
        self.assertAlmostEqual(popularity["percentage"].sum(), 1.0)

    @given(strategies.interactions())
    def test_generated_percentage_is_the_share_of_interactions(self, data) -> None:
        popularity = PopularityPercentage().segment(data)
        expected = data["item"].value_counts(normalize=True).to_dict()
        for item, share in percentages(popularity, "item").items():
            with self.subTest(item=item):
                self.assertAlmostEqual(share, expected[item])


class DiscreteFeatureSegmentationTest(TestCase):
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
        with self.assertRaisesMessage(
            InvalidValueException,
            "Feature contains -1 as value. Please select another fill_na value.",
        ):
            DiscreteFeatureSegmentation().segment(feature=user_ages())

    @given(strategies.discrete_features(key="item"))
    def test_entities_with_the_same_value_share_a_group(self, feature) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=feature.copy())
        values = feature["feature"].fillna("missing")
        for value in values.unique():
            with self.subTest(value=value):
                self.assertEqual(segmented.loc[values == value, "group"].nunique(), 1)

    @given(strategies.discrete_features(key="item"))
    def test_entities_with_different_values_get_different_groups(self, feature) -> None:
        segmented = DiscreteFeatureSegmentation().segment(feature=feature.copy())
        values = feature["feature"].fillna("missing")
        self.assertEqual(segmented["group"].nunique(), values.nunique())


class FindRelevantItemsTest(TestCase):
    @given(strategies.interactions())
    def test_lists_every_item_each_user_interacted_with(self, data) -> None:
        relevant = find_relevant_items(data)
        for user, items in zip(relevant["user"], relevant["pos_items"]):
            with self.subTest(user=user):
                self.assertEqual(
                    set(items), set(data.loc[data["user"] == user, "item"])
                )

    @given(strategies.interactions())
    def test_has_one_row_per_user(self, data) -> None:
        relevant = find_relevant_items(data)
        self.assertEqual(sorted(relevant["user"]), sorted(data["user"].unique()))


if __name__ == "__main__":
    unittest.main()
