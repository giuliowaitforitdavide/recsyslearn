import unittest

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal, assert_series_equal

from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.errors.metrics import (
    DivisionByZeroException,
    InvalidCutoffException,
    RecListTooShortException,
)
from recsyslearn.result import MetricResult
from tests import strategies
from tests.case import TestCase
from tests.fixtures import pos_items, top_5, top_6


def dcg(gains: list[int]) -> float:
    """DCG with binary gains in ranking order: sum of (2^rel - 1) / ln(position + 1)."""
    return sum(
        (2**gain - 1) / np.log(position + 1) for position, gain in enumerate(gains, 1)
    )


@st.composite
def lists_and_relevant_items(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Generated recommendation lists, relevant items of every user and a computable cut-off."""
    items = draw(strategies.catalogues())
    top_n = draw(strategies.top_n(items=items))
    users = sorted(top_n["user"].unique())
    relevant = draw(strategies.pos_items(users, items))
    k = draw(
        st.integers(min_value=1, max_value=int(top_n.groupby("user").size().min()))
    )
    return top_n, relevant, k


class NDCGTest(TestCase):
    def test_matches_the_formula_at_cutoff_2(self) -> None:
        """NDCG@2 = DCG@2 / IDCG@2, with gain 2^rel - 1 and discount ln(position + 1)."""
        result = NDCG(k=2).evaluate(top_5(), pos_items())
        expected = pd.Series(
            {
                "1": dcg([1, 0]) / dcg([1, 1]),
                "2": dcg([0, 0]) / dcg([1, 1]),
                "3": dcg([0, 1]) / dcg([1, 1]),
                "4": dcg([0, 1]) / dcg([1, 1]),
                "5": dcg([1, 1]) / dcg([1, 1]),
                "6": dcg([1, 0]) / dcg([1]),
            },
            name="NDCG@2",
        ).rename_axis("user")
        assert_series_equal(result.per_user, expected)

    def test_aggregate_is_the_mean_over_users(self) -> None:
        result = NDCG(k=2).evaluate(top_5(), pos_items())
        self.assertAlmostEqual(result.aggregate, result.per_user.mean())

    def test_returns_a_metric_result_named_after_the_cutoff(self) -> None:
        result = NDCG(k=2).evaluate(top_5(), pos_items())
        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "NDCG@2")

    def test_has_one_float64_value_per_user(self) -> None:
        per_user = NDCG(k=2).evaluate(top_5(), pos_items()).per_user
        self.assertEqual(list(per_user.index), ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(per_user.dtype, np.float64)

    def test_orders_each_list_by_its_rank(self) -> None:
        """
        In top_6 the rows are not in rank order: user 1 has item 9 at rank 2 but in row 6.

        Ranked list of user 1: 1, 9, …; relevant items 2 and 9, so NDCG@2 = DCG([0, 1]) / DCG([1, 1]).
        """
        per_user = NDCG(k=2).evaluate(top_6(), pos_items()).per_user
        self.assertAlmostEqual(per_user["1"], dcg([0, 1]) / dcg([1, 1]))

    def test_requires_the_cutoff(self) -> None:
        with self.assertRaisesMessage(
            TypeError, "NDCG.__init__() missing 1 required positional argument: 'k'"
        ):
            NDCG()

    def test_raises_for_a_cutoff_that_is_not_a_positive_integer(self) -> None:
        for k, shown in [(0, "0"), (-1, "-1"), (2.5, "2.5"), (True, "True")]:
            with self.subTest(k=k):
                with self.assertRaisesMessage(
                    InvalidCutoffException,
                    f"NDCG: the cut-off k must be a positive integer, got {shown}.",
                ):
                    NDCG(k=k)

    def test_raises_when_the_cutoff_is_longer_than_the_shortest_list(self) -> None:
        with self.assertRaisesMessage(
            RecListTooShortException,
            "NDCG@3 needs recommendation lists of at least 3 items: the shortest has 2.",
        ):
            NDCG(k=3).evaluate(top_5().iloc[[0, 1]], pos_items())

    def test_raises_for_user_without_relevant_items(self) -> None:
        """The ideal DCG of a user without relevant items is zero: NDCG is undefined (#70)."""
        with self.assertRaisesMessage(
            DivisionByZeroException, "NDCG: division by zero for user 1."
        ):
            NDCG(k=2).evaluate(self._user_1_lists(), self._no_relevant_items())

    def test_does_not_mutate_the_input(self) -> None:
        top_n, relevant = top_6(), pos_items()
        NDCG(k=2).evaluate(top_n, relevant)
        assert_frame_equal(top_n, top_6())
        assert_frame_equal(relevant, pos_items())

    @given(lists_and_relevant_items())
    def test_is_between_zero_and_one(self, data) -> None:
        top_n, relevant, k = data
        per_user = NDCG(k=k).evaluate(top_n, relevant).per_user
        self.assertTrue(((per_user >= 0) & (per_user <= 1 + 1e-12)).all())

    @given(lists_and_relevant_items())
    def test_is_one_for_a_perfect_ranking(self, data) -> None:
        """Lists that start with all the relevant items (or are made only of them) have NDCG 1."""
        top_n, _, k = data
        relevant = (
            top_n[top_n["rank"] <= k]
            .groupby("user")["item"]
            .apply(np.asarray)
            .reset_index()
        ).rename(columns={"item": "pos_items"})
        per_user = NDCG(k=k).evaluate(top_n, relevant).per_user
        np.testing.assert_allclose(per_user.to_numpy(), 1.0)

    @given(lists_and_relevant_items(), st.randoms())
    def test_row_order_does_not_change_the_result(self, data, random) -> None:
        top_n, relevant, k = data
        shuffled = top_n.sample(frac=1, random_state=random.randint(0, 2**32 - 1))
        assert_series_equal(
            NDCG(k=k).evaluate(shuffled, relevant).per_user,
            NDCG(k=k).evaluate(top_n, relevant).per_user,
        )

    @given(lists_and_relevant_items())
    def test_does_not_mutate_generated_input(self, data) -> None:
        top_n, relevant, k = data
        original_top_n, original_relevant = top_n.copy(), relevant.copy()
        NDCG(k=k).evaluate(top_n, relevant)
        assert_frame_equal(top_n, original_top_n)
        assert_frame_equal(relevant, original_relevant)

    @staticmethod
    def _user_1_lists() -> pd.DataFrame:
        """The top-5 list of user 1 only."""
        return top_5().query("user == '1'")

    @staticmethod
    def _no_relevant_items() -> pd.DataFrame:
        """User 1 with an empty list of relevant items."""
        return pd.DataFrame({"user": ["1"], "pos_items": [np.array([], dtype=object)]})


if __name__ == "__main__":
    unittest.main()
