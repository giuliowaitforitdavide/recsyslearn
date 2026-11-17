import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from recsyslearn.fairness.metrics import Entropy, KullbackLeibler, MutualInformation
from recsyslearn.fairness.utils import exp_matrix, prob_matrix
from tests.utils import (
    first_example,
    item_groups,
    rel_matrix_1,
    second_example,
    user_groups,
)


class EntropyTest(unittest.TestCase):
    def test_user_exposure(self) -> None:
        top_n = first_example.merge(user_groups, on="user")
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 0.91830, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on="item")
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 1.48547, delta=1e-5)


class KullbackLeiblerTest(unittest.TestCase):
    def test_user_effectiveness(self) -> None:
        top_n = second_example.merge(user_groups, on="user")
        rel_matrix = rel_matrix_1.merge(user_groups, on="user")
        target_representation = pd.DataFrame(
            [["1", 0.5], ["2", 0.5]], columns=["group", "target_representation"]
        )
        divergence = KullbackLeibler().evaluate(
            top_n, target_representation, rel_matrix
        )
        self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on="item")
        target_representation = pd.DataFrame(
            [["1", 0.2], ["2", 0.3], ["3", 0.5]],
            columns=["group", "target_representation"],
        )
        divergence = KullbackLeibler().evaluate(top_n, target_representation)
        self.assertAlmostEqual(divergence, 0, delta=1e-5)


class MutualInformationTest(unittest.TestCase):
    def test_user_exposure(self) -> None:
        top_n = first_example.merge(user_groups, on="user")
        mi = MutualInformation().evaluate(top_n, "user")
        self.assertAlmostEqual(mi, 0.25582, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on="item")
        mi = MutualInformation().evaluate(top_n, "item")
        self.assertAlmostEqual(mi, 0.10570, delta=1e-5)

    def test_flag_not_valid(self) -> None:
        with self.assertRaises(KeyError) as context:
            top_n = first_example.merge(item_groups, on="item")
            MutualInformation().evaluate(top_n, "ratings")

        self.assertIn("Invalid flag 'ratings'", str(context.exception))


class InputNotMutatedTest(unittest.TestCase):
    """Fairness metrics and helpers leave the caller's DataFrames unchanged (#75)."""

    def setUp(self) -> None:
        self.top_n = first_example.merge(item_groups, on="item")
        self.top_n_before = self.top_n.copy()
        self.target_representation = pd.DataFrame(
            [["1", 0.2], ["2", 0.3], ["3", 0.5]],
            columns=["group", "target_representation"],
        )

    def assert_input_unchanged_and_repeatable(self, evaluate) -> None:
        first = evaluate(self.top_n)
        second = evaluate(self.top_n)

        assert_frame_equal(self.top_n, self.top_n_before)
        self.assertEqual(first, second)

    def test_exp_matrix(self) -> None:
        exp_matrix(self.top_n)
        assert_frame_equal(self.top_n, self.top_n_before)

    def test_prob_matrix(self) -> None:
        prob_matrix(self.top_n)
        assert_frame_equal(self.top_n, self.top_n_before)

    def test_prob_matrix_float_rank(self) -> None:
        """A float `rank` column keeps its dtype, so it must not be overwritten in place."""
        top_n = self.top_n.astype({"rank": float})
        top_n_before = top_n.copy()
        prob_matrix(top_n)
        assert_frame_equal(top_n, top_n_before)

    def test_entropy(self) -> None:
        self.assert_input_unchanged_and_repeatable(
            lambda top_n: Entropy().evaluate(top_n)
        )

    def test_kullback_leibler(self) -> None:
        self.assert_input_unchanged_and_repeatable(
            lambda top_n: KullbackLeibler().evaluate(top_n, self.target_representation)
        )

    def test_kullback_leibler_effectiveness(self) -> None:
        """Effectiveness path: the relevance matrix is left unchanged too."""
        top_n = second_example.merge(user_groups, on="user")
        top_n_before = top_n.copy()
        rel_matrix = rel_matrix_1.merge(user_groups, on="user")
        rel_matrix_before = rel_matrix.copy()
        target_representation = pd.DataFrame(
            [["1", 0.5], ["2", 0.5]], columns=["group", "target_representation"]
        )

        first = KullbackLeibler().evaluate(top_n, target_representation, rel_matrix)
        second = KullbackLeibler().evaluate(top_n, target_representation, rel_matrix)

        assert_frame_equal(top_n, top_n_before)
        assert_frame_equal(rel_matrix, rel_matrix_before)
        self.assertEqual(first, second)

    def test_mutual_information(self) -> None:
        self.assert_input_unchanged_and_repeatable(
            lambda top_n: MutualInformation().evaluate(top_n, "item")
        )


if __name__ == "__main__":
    unittest.main()
