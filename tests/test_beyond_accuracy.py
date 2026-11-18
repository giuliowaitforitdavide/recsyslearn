import unittest

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal, assert_series_equal

from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from recsyslearn.errors.metrics import DivisionByZeroException
from recsyslearn.result import MetricResult
from tests import strategies
from tests.case import TestCase
from tests.fixtures import item_groups, item_popularity, top_5_unranked, top_6


@st.composite
def lists_and_catalogue(draw: st.DrawFn) -> tuple[pd.DataFrame, list[str]]:
    """Generated recommendation lists and the catalogue their items come from."""
    items = draw(strategies.catalogues())
    return draw(strategies.top_n(items=items)), items


@st.composite
def lists_with_popularity(draw: st.DrawFn) -> pd.DataFrame:
    """Generated recommendation lists with a popularity in (0, 1] for each item."""
    items = draw(strategies.catalogues())
    popularity = pd.DataFrame(
        {
            "item": items,
            "percentage": [
                draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False))
                for _ in items
            ],
        }
    )
    return draw(strategies.top_n(items=items)).merge(popularity, on="item")


class CoverageTest(TestCase):
    def test_seven_of_ten_items_recommended(self) -> None:
        top_n = top_5_unranked()
        top_n = top_n[~top_n["item"].isin(["1", "3", "4"])]
        cov = Coverage().evaluate(top_n, item_groups().item.tolist())
        self.assertAlmostEqual(cov.aggregate, 0.7)

    def test_four_of_ten_items_recommended(self) -> None:
        top_n = top_5_unranked()
        top_n = top_n[~top_n["item"].isin(["1", "2", "4", "7", "9", "10"])]
        cov = Coverage().evaluate(top_n, item_groups().item.tolist())
        self.assertAlmostEqual(cov.aggregate, 0.4)

    def test_is_a_population_level_result(self) -> None:
        """Coverage is defined over all the lists together: it has no per-user values."""
        result = Coverage().evaluate(top_5_unranked(), item_groups().item.tolist())
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "Coverage")
        self.assertIsNone(result.per_user)

    def test_raises_for_an_empty_catalogue(self) -> None:
        """Coverage is undefined for an empty item catalogue (#70)."""
        with self.assertRaisesMessage(
            DivisionByZeroException, "Coverage: division by zero."
        ):
            Coverage().evaluate(top_5_unranked(), [])

    @given(lists_and_catalogue())
    def test_is_between_zero_and_one(self, data) -> None:
        top_n, items = data
        coverage = Coverage().evaluate(top_n, items).aggregate
        self.assertTrue(0 < coverage <= 1)

    @given(lists_and_catalogue(), st.randoms())
    def test_row_order_does_not_change_the_result(self, data, random) -> None:
        top_n, items = data
        shuffled = top_n.sample(frac=1, random_state=random.randint(0, 2**32 - 1))
        self.assertEqual(
            Coverage().evaluate(shuffled, items).aggregate,
            Coverage().evaluate(top_n, items).aggregate,
        )

    @given(lists_and_catalogue())
    def test_does_not_mutate_the_input(self, data) -> None:
        top_n, items = data
        original = top_n.copy()
        Coverage().evaluate(top_n, items)
        assert_frame_equal(top_n, original)


class NoveltyTest(TestCase):
    def test_uses_the_group_column_as_popularity_by_default(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(
            nov.aggregate, self._expected(top_n["group"]), delta=1e-5
        )

    def test_does_not_depend_on_the_rank(self) -> None:
        """top_6 has different ranks per item; the expected value ignores them."""
        top_n = top_6().merge(item_groups(), on="item")
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(
            nov.aggregate, self._expected(top_n["group"]), delta=1e-5
        )

    def test_uses_the_given_popularity_column(self) -> None:
        top_n = top_5_unranked().merge(item_popularity(), on="item")
        nov = Novelty().evaluate(top_n, popularity_definition="percentage")
        self.assertAlmostEqual(
            nov.aggregate, self._expected(top_n["percentage"]), delta=1e-5
        )

    def test_per_user_value_is_the_mean_over_the_list(self) -> None:
        """Novelty of a user = mean of -log2(popularity) over their list."""
        top_n = top_5_unranked().merge(item_popularity(), on="item")
        per_user = Novelty().evaluate(top_n, "percentage").per_user
        for user, user_list in top_n.groupby("user"):
            with self.subTest(user=user):
                self.assertAlmostEqual(
                    per_user[user], self._expected(user_list["percentage"])
                )

    def test_returns_a_per_user_metric_result(self) -> None:
        result = Novelty().evaluate(top_5_unranked().merge(item_groups(), on="item"))
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "Novelty")
        self.assertEqual(list(result.per_user.index), ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(result.per_user.name, "Novelty")

    def test_aggregate_is_the_mean_over_users(self) -> None:
        result = Novelty().evaluate(top_5_unranked().merge(item_groups(), on="item"))
        self.assertAlmostEqual(result.aggregate, result.per_user.mean())

    def test_does_not_mutate_the_input(self) -> None:
        """Regression test: the popularity column of the input was converted in place."""
        top_n = top_5_unranked().merge(item_groups(), on="item")
        Novelty().evaluate(top_n)
        assert_frame_equal(top_n, top_5_unranked().merge(item_groups(), on="item"))

    @given(lists_with_popularity())
    def test_is_non_negative(self, top_n) -> None:
        """Popularity is at most 1, so -log2(popularity) >= 0."""
        per_user = Novelty().evaluate(top_n, "percentage").per_user
        self.assertTrue((per_user >= 0).all())

    @given(lists_with_popularity(), st.randoms())
    def test_row_order_does_not_change_the_result(self, top_n, random) -> None:
        shuffled = top_n.sample(frac=1, random_state=random.randint(0, 2**32 - 1))
        assert_series_equal(
            Novelty().evaluate(shuffled, "percentage").per_user,
            Novelty().evaluate(top_n, "percentage").per_user,
        )

    @given(lists_with_popularity())
    def test_does_not_mutate_generated_input(self, top_n) -> None:
        original = top_n.copy()
        Novelty().evaluate(top_n, "percentage")
        assert_frame_equal(top_n, original)

    @staticmethod
    def _expected(popularity: pd.Series) -> float:
        """Mean of -log2(popularity) over all recommended items."""
        return float(np.mean(-np.log2(popularity.astype(float).to_numpy())))


if __name__ == "__main__":
    unittest.main()
