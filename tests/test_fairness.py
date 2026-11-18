import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from recsyslearn.errors.metrics import DivisionByZeroException
from recsyslearn.fairness.metrics import Entropy, KullbackLeibler, MutualInformation
from recsyslearn.fairness.utils import eff_matrix, exp_matrix, prob_matrix
from tests.fixtures import (
    binary_relevance,
    graded_relevance,
    item_groups,
    top_5_unranked,
    top_6,
    user_groups,
)


class EntropyTest(unittest.TestCase):
    def test_user_exposure(self) -> None:
        top_n = top_5_unranked().merge(user_groups(), on="user")
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 0.91830, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 1.48547, delta=1e-5)


class KullbackLeiblerTest(unittest.TestCase):
    def test_user_effectiveness(self) -> None:
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = binary_relevance().merge(user_groups(), on="user")
        target_representation = pd.DataFrame(
            [["1", 0.5], ["2", 0.5]], columns=["group", "target_representation"]
        )
        divergence = KullbackLeibler().evaluate(
            top_n, target_representation, rel_matrix
        )
        self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        target_representation = pd.DataFrame(
            [["1", 0.2], ["2", 0.3], ["3", 0.5]],
            columns=["group", "target_representation"],
        )
        divergence = KullbackLeibler().evaluate(top_n, target_representation)
        self.assertAlmostEqual(divergence, 0, delta=1e-5)


class MutualInformationTest(unittest.TestCase):
    def test_user_exposure(self) -> None:
        top_n = top_5_unranked().merge(user_groups(), on="user")
        mi = MutualInformation().evaluate(top_n, "user")
        self.assertAlmostEqual(mi, 0.25582, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        mi = MutualInformation().evaluate(top_n, "item")
        self.assertAlmostEqual(mi, 0.10570, delta=1e-5)

    def test_flag_not_valid(self) -> None:
        with self.assertRaises(KeyError) as context:
            top_n = top_5_unranked().merge(item_groups(), on="item")
            MutualInformation().evaluate(top_n, "ratings")

        self.assertIn("Invalid flag 'ratings'", str(context.exception))


class DivisionByZeroTest(unittest.TestCase):
    """Undefined divisions in fairness metrics raise DivisionByZeroException (#70)."""

    def test_entropy_zero_total_exposure(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item").assign(rank=0)
        with self.assertRaises(DivisionByZeroException) as context:
            Entropy().evaluate(top_n)
        self.assertEqual(str(context.exception), "Entropy: division by zero.")

    def test_kullback_leibler_zero_target(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        target_representation = pd.DataFrame(
            [["1", 0.5], ["2", 0.5], ["3", 0.0]],
            columns=["group", "target_representation"],
        )
        with self.assertRaises(DivisionByZeroException) as context:
            KullbackLeibler().evaluate(top_n, target_representation)
        self.assertEqual(
            str(context.exception), "KullbackLeibler: division by zero for group 3."
        )

    def test_kullback_leibler_zero_rank(self) -> None:
        """Exposure 1 / log2(1 + rank) is undefined for rank 0 (ranks start at 1)."""
        top_n = top_5_unranked().merge(item_groups(), on="item").assign(rank=0)
        target_representation = pd.DataFrame(
            [["1", 0.2], ["2", 0.3], ["3", 0.5]],
            columns=["group", "target_representation"],
        )
        with self.assertRaises(DivisionByZeroException) as context:
            KullbackLeibler().evaluate(top_n, target_representation)
        self.assertTrue(
            str(context.exception).startswith("KullbackLeibler: division by zero")
        )

    def test_mutual_information_no_relevant_item_recommended(self) -> None:
        """With effectiveness, no relevant item recommended gives zero total exposure."""
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = (
            binary_relevance()
            .assign(item="not recommended")
            .merge(user_groups(), on="user")
        )
        with self.assertRaises(DivisionByZeroException) as context:
            MutualInformation().evaluate(top_n, "user", rel_matrix)
        self.assertEqual(str(context.exception), "MutualInformation: division by zero.")

    def test_zero_probability_terms_are_skipped(self) -> None:
        """
        A group with zero probability contributes 0 (convention 0 · log(0 / q) = 0), it does not raise.

        All ranks are 1, so every recommended item has exposure 1. Only the 6 items of group 1
        and the 9 items of group 2 are relevant, so p = (0.4, 0.6, 0) and
        KL = 0.4 · log2(0.4 / 0.2) + 0.6 · log2(0.6 / 0.3) + 0 = 1.
        """
        top_n = top_5_unranked().merge(item_groups(), on="item")
        rel_matrix = top_n[top_n["group"] != "3"].assign(rank=1)
        target_representation = pd.DataFrame(
            [["1", 0.2], ["2", 0.3], ["3", 0.5]],
            columns=["group", "target_representation"],
        )
        divergence = KullbackLeibler().evaluate(
            top_n, target_representation, rel_matrix
        )
        self.assertAlmostEqual(divergence, 1.0)
        mi = MutualInformation().evaluate(top_n, "item", rel_matrix)
        self.assertTrue(np.isfinite(mi))


class InputNotMutatedTest(unittest.TestCase):
    """Fairness metrics and helpers leave the caller's DataFrames unchanged (#75)."""

    def setUp(self) -> None:
        self.top_n = top_5_unranked().merge(item_groups(), on="item")
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
        top_n = top_6().merge(user_groups(), on="user")
        top_n_before = top_n.copy()
        rel_matrix = binary_relevance().merge(user_groups(), on="user")
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


def assert_frame_equal_unordered(
    left: pd.DataFrame, right: pd.DataFrame, **kwargs
) -> None:
    """
    Compare DataFrames regardless of row order, since merge row order changed in pandas 2.2.
    """

    def sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["user", "item"]).reset_index(drop=True)

    assert_frame_equal(sort(left), sort(right), **kwargs)


class ExpMatrixTest(unittest.TestCase):
    """
    Tester for the exp_matrix function.
    """

    def test_exp_matrix_1(self) -> None:
        top_n = top_5_unranked().merge(user_groups(), on=["user"])
        result = exp_matrix(top_n)
        assert_frame_equal(
            result,
            pd.DataFrame(
                [
                    ["1", "3", 1.0, "1"],
                    ["1", "4", 1.0, "1"],
                    ["1", "6", 1.0, "1"],
                    ["1", "7", 1.0, "1"],
                    ["1", "8", 1.0, "1"],
                    ["2", "1", 1.0, "1"],
                    ["2", "2", 1.0, "1"],
                    ["2", "5", 1.0, "1"],
                    ["2", "6", 1.0, "1"],
                    ["2", "7", 1.0, "1"],
                    ["3", "2", 1.0, "2"],
                    ["3", "3", 1.0, "2"],
                    ["3", "6", 1.0, "2"],
                    ["3", "9", 1.0, "2"],
                    ["3", "10", 1.0, "2"],
                    ["4", "1", 1.0, "2"],
                    ["4", "3", 1.0, "2"],
                    ["4", "6", 1.0, "2"],
                    ["4", "7", 1.0, "2"],
                    ["4", "9", 1.0, "2"],
                    ["5", "1", 1.0, "2"],
                    ["5", "3", 1.0, "2"],
                    ["5", "5", 1.0, "2"],
                    ["5", "7", 1.0, "2"],
                    ["5", "9", 1.0, "2"],
                    ["6", "2", 1.0, "2"],
                    ["6", "3", 1.0, "2"],
                    ["6", "5", 1.0, "2"],
                    ["6", "9", 1.0, "2"],
                    ["6", "10", 1.0, "2"],
                ],
                columns=["user", "item", "rank", "group"],
            ),
            check_like=True,
            check_exact=False,
            rtol=1e-3,
        )

    def test_exp_matrix_2(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on=["item"])
        result = exp_matrix(top_n)
        assert_frame_equal_unordered(
            result,
            pd.DataFrame(
                [
                    ["1", "3", 1.0, "2"],
                    ["3", "3", 1.0, "2"],
                    ["4", "3", 1.0, "2"],
                    ["5", "3", 1.0, "2"],
                    ["6", "3", 1.0, "2"],
                    ["1", "4", 1.0, "2"],
                    ["1", "6", 1.0, "3"],
                    ["2", "6", 1.0, "3"],
                    ["3", "6", 1.0, "3"],
                    ["4", "6", 1.0, "3"],
                    ["1", "7", 1.0, "3"],
                    ["2", "7", 1.0, "3"],
                    ["4", "7", 1.0, "3"],
                    ["5", "7", 1.0, "3"],
                    ["1", "8", 1.0, "3"],
                    ["2", "1", 1.0, "1"],
                    ["4", "1", 1.0, "1"],
                    ["5", "1", 1.0, "1"],
                    ["2", "2", 1.0, "1"],
                    ["3", "2", 1.0, "1"],
                    ["6", "2", 1.0, "1"],
                    ["2", "5", 1.0, "2"],
                    ["5", "5", 1.0, "2"],
                    ["6", "5", 1.0, "2"],
                    ["3", "9", 1.0, "3"],
                    ["4", "9", 1.0, "3"],
                    ["5", "9", 1.0, "3"],
                    ["6", "9", 1.0, "3"],
                    ["3", "10", 1.0, "3"],
                    ["6", "10", 1.0, "3"],
                ],
                columns=["user", "item", "rank", "group"],
            ),
            check_like=True,
            check_exact=False,
            rtol=1e-3,
        )


class ProbMatrixTest(unittest.TestCase):
    """
    Tester for the prob_matrix function.
    """

    def test_prob_matrix_1(self) -> None:
        top_n = top_5_unranked().merge(user_groups(), on=["user"])
        result = prob_matrix(top_n)
        assert_frame_equal(
            result,
            pd.DataFrame(
                [
                    ["1", "3", 0.03333333333333333, "1"],
                    ["1", "4", 0.03333333333333333, "1"],
                    ["1", "6", 0.03333333333333333, "1"],
                    ["1", "7", 0.03333333333333333, "1"],
                    ["1", "8", 0.03333333333333333, "1"],
                    ["2", "1", 0.03333333333333333, "1"],
                    ["2", "2", 0.03333333333333333, "1"],
                    ["2", "5", 0.03333333333333333, "1"],
                    ["2", "6", 0.03333333333333333, "1"],
                    ["2", "7", 0.03333333333333333, "1"],
                    ["3", "2", 0.03333333333333333, "2"],
                    ["3", "3", 0.03333333333333333, "2"],
                    ["3", "6", 0.03333333333333333, "2"],
                    ["3", "9", 0.03333333333333333, "2"],
                    ["3", "10", 0.03333333333333333, "2"],
                    ["4", "1", 0.03333333333333333, "2"],
                    ["4", "3", 0.03333333333333333, "2"],
                    ["4", "6", 0.03333333333333333, "2"],
                    ["4", "7", 0.03333333333333333, "2"],
                    ["4", "9", 0.03333333333333333, "2"],
                    ["5", "1", 0.03333333333333333, "2"],
                    ["5", "3", 0.03333333333333333, "2"],
                    ["5", "5", 0.03333333333333333, "2"],
                    ["5", "7", 0.03333333333333333, "2"],
                    ["5", "9", 0.03333333333333333, "2"],
                    ["6", "2", 0.03333333333333333, "2"],
                    ["6", "3", 0.03333333333333333, "2"],
                    ["6", "5", 0.03333333333333333, "2"],
                    ["6", "9", 0.03333333333333333, "2"],
                    ["6", "10", 0.03333333333333333, "2"],
                ],
                columns=["user", "item", "rank", "group"],
            ),
            check_like=True,
            check_exact=False,
            rtol=1e-3,
        )

    def test_prob_matrix_2(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on=["item"])
        result = prob_matrix(top_n)
        assert_frame_equal_unordered(
            result,
            pd.DataFrame(
                [
                    ["1", "3", 0.03333333333333333, "2"],
                    ["3", "3", 0.03333333333333333, "2"],
                    ["4", "3", 0.03333333333333333, "2"],
                    ["5", "3", 0.03333333333333333, "2"],
                    ["6", "3", 0.03333333333333333, "2"],
                    ["1", "4", 0.03333333333333333, "2"],
                    ["1", "6", 0.03333333333333333, "3"],
                    ["2", "6", 0.03333333333333333, "3"],
                    ["3", "6", 0.03333333333333333, "3"],
                    ["4", "6", 0.03333333333333333, "3"],
                    ["1", "7", 0.03333333333333333, "3"],
                    ["2", "7", 0.03333333333333333, "3"],
                    ["4", "7", 0.03333333333333333, "3"],
                    ["5", "7", 0.03333333333333333, "3"],
                    ["1", "8", 0.03333333333333333, "3"],
                    ["2", "1", 0.03333333333333333, "1"],
                    ["4", "1", 0.03333333333333333, "1"],
                    ["5", "1", 0.03333333333333333, "1"],
                    ["2", "2", 0.03333333333333333, "1"],
                    ["3", "2", 0.03333333333333333, "1"],
                    ["6", "2", 0.03333333333333333, "1"],
                    ["2", "5", 0.03333333333333333, "2"],
                    ["5", "5", 0.03333333333333333, "2"],
                    ["6", "5", 0.03333333333333333, "2"],
                    ["3", "9", 0.03333333333333333, "3"],
                    ["4", "9", 0.03333333333333333, "3"],
                    ["5", "9", 0.03333333333333333, "3"],
                    ["6", "9", 0.03333333333333333, "3"],
                    ["3", "10", 0.03333333333333333, "3"],
                    ["6", "10", 0.03333333333333333, "3"],
                ],
                columns=["user", "item", "rank", "group"],
            ),
            check_like=True,
            check_exact=False,
            rtol=1e-3,
        )


class EffMatrixTest(unittest.TestCase):
    def test_eff_matrix_1(self) -> None:
        top_n = top_6().merge(user_groups(), on=["user"])
        rel_matrix = binary_relevance().merge(user_groups(), on=["user"])
        result = eff_matrix(top_n, rel_matrix)
        result = result[result["rank"] != 0].reset_index(drop=True)
        assert_frame_equal(
            result,
            pd.DataFrame(
                [
                    ["1", "9", 0.6309297536, "1"],
                    ["2", "6", 0.4306765581, "1"],
                    ["2", "7", 0.5, "1"],
                    ["3", "7", 0.4306765581, "2"],
                    ["3", "9", 0.3562071871, "2"],
                    ["4", "3", 0.3868528072, "2"],
                    ["4", "7", 0.5, "2"],
                    ["6", "2", 0.5, "2"],
                    ["6", "9", 1, "2"],
                ],
                columns=["user", "item", "rank", "group"],
            ),
            check_like=True,
            check_exact=False,
            rtol=1e-3,
        )

    def test_eff_matrix_2(self) -> None:
        top_n = top_6().merge(item_groups(), on=["item"])
        rel_matrix = graded_relevance().merge(item_groups(), on=["item"])
        result = eff_matrix(top_n, rel_matrix)
        result = result[result["rank"] != 0].reset_index(drop=True)
        assert_frame_equal_unordered(
            result,
            pd.DataFrame(
                [
                    ["5", "1", 0.05088674102, "1"],
                    ["3", "3", 0.09013282194, "2"],
                    ["4", "3", 0.05526468675, "2"],
                    ["3", "4", 0.07142857143, "2"],
                    ["5", "5", 0.07142857143, "2"],
                    ["4", "8", 0.04853305797, "3"],
                    ["6", "8", 0.04853305797, "3"],
                    ["1", "9", 0.04853305797, "3"],
                    ["3", "9", 0.02740055285, "3"],
                    ["4", "9", 0.07692307692, "3"],
                    ["5", "9", 0.07692307692, "3"],
                    ["6", "9", 0.07692307692, "3"],
                    ["6", "2", 0.07142857143, "1"],
                    ["2", "6", 0.03312896601, "3"],
                    ["5", "6", 0.03312896601, "3"],
                    ["6", "6", 0.02740055285, "3"],
                    ["2", "7", 0.03846153846, "3"],
                    ["3", "7", 0.03312896601, "3"],
                    ["4", "7", 0.03846153846, "3"],
                ],
                columns=["user", "item", "rank", "group"],
            ),
            check_like=True,
            check_exact=False,
            rtol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
