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


def item_exposure_lists() -> pd.DataFrame:
    """top_5_unranked with the group of each item: every item has rank 1, so exposure 1."""
    return top_5_unranked().merge(item_groups(), on="item")


def user_exposure_lists() -> pd.DataFrame:
    """top_5_unranked with the group of each user."""
    return top_5_unranked().merge(user_groups(), on="user")


def item_target(shares: tuple[float, float, float] = (0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Target representation of the three item groups."""
    return pd.DataFrame(
        {"group": ["1", "2", "3"], "target_representation": list(shares)}
    )


def user_target() -> pd.DataFrame:
    """Equal target representation of the two user groups."""
    return pd.DataFrame({"group": ["1", "2"], "target_representation": [0.5, 0.5]})


def relevant_except_group_3() -> pd.DataFrame:
    """Relevance 1 for the recommended items of groups 1 and 2, none for group 3."""
    top_n = item_exposure_lists()
    return top_n[top_n["group"] != "3"].assign(rank=1)


def assert_frame_equal_unordered(
    left: pd.DataFrame, right: pd.DataFrame, **kwargs
) -> None:
    """
    Compare DataFrames regardless of row order, since merge row order changed in pandas 2.2.
    """

    def sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["user", "item"]).reset_index(drop=True)

    assert_frame_equal(sort(left), sort(right), **kwargs)


class EntropyTest(unittest.TestCase):
    def test_entropy_of_user_group_exposure(self) -> None:
        self.assertAlmostEqual(
            Entropy().evaluate(user_exposure_lists()), 0.91830, delta=1e-5
        )

    def test_entropy_of_item_group_exposure(self) -> None:
        self.assertAlmostEqual(
            Entropy().evaluate(item_exposure_lists()), 1.48547, delta=1e-5
        )

    def test_raises_when_the_total_exposure_is_zero(self) -> None:
        """A distribution over groups is undefined when the total exposure is zero (#70)."""
        with self.assertRaises(DivisionByZeroException):
            Entropy().evaluate(item_exposure_lists().assign(rank=0))

    def test_division_error_names_the_metric(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            Entropy().evaluate(item_exposure_lists().assign(rank=0))

        self.assertEqual(str(context.exception), "Entropy: division by zero.")

    def test_does_not_mutate_the_input(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        Entropy().evaluate(top_n)
        assert_frame_equal(top_n, item_exposure_lists())

    def test_repeated_calls_give_the_same_value(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        self.assertEqual(Entropy().evaluate(top_n), Entropy().evaluate(top_n))


class KullbackLeiblerTest(unittest.TestCase):
    def test_divergence_of_user_group_effectiveness(self) -> None:
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = binary_relevance().merge(user_groups(), on="user")
        divergence = KullbackLeibler().evaluate(top_n, user_target(), rel_matrix)
        self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)

    def test_zero_when_the_item_exposure_matches_the_target(self) -> None:
        divergence = KullbackLeibler().evaluate(item_exposure_lists(), item_target())
        self.assertAlmostEqual(divergence, 0, delta=1e-5)

    def test_groups_with_zero_probability_contribute_zero(self) -> None:
        """
        A group with zero probability contributes 0 (convention 0 · log(0 / q) = 0), it does not raise.

        All ranks are 1, so every recommended item has exposure 1. Only the 6 items of group 1
        and the 9 items of group 2 are relevant, so p = (0.4, 0.6, 0) and
        KL = 0.4 · log2(0.4 / 0.2) + 0.6 · log2(0.6 / 0.3) + 0 = 1.
        """
        divergence = KullbackLeibler().evaluate(
            item_exposure_lists(), item_target(), relevant_except_group_3()
        )
        self.assertAlmostEqual(divergence, 1.0)

    def test_raises_for_a_zero_target_on_an_exposed_group(self) -> None:
        """The divergence is infinite when an exposed group has a target of 0 (#70)."""
        with self.assertRaises(DivisionByZeroException):
            KullbackLeibler().evaluate(
                item_exposure_lists(), item_target((0.5, 0.5, 0))
            )

    def test_division_error_names_the_group(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            KullbackLeibler().evaluate(
                item_exposure_lists(), item_target((0.5, 0.5, 0))
            )

        self.assertEqual(
            str(context.exception), "KullbackLeibler: division by zero for group 3."
        )

    def test_raises_for_rank_zero(self) -> None:
        """Exposure 1 / log2(1 + rank) is undefined for rank 0 (ranks start at 1)."""
        with self.assertRaises(DivisionByZeroException):
            KullbackLeibler().evaluate(
                item_exposure_lists().assign(rank=0), item_target()
            )

    def test_does_not_mutate_the_input(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        KullbackLeibler().evaluate(top_n, item_target())
        assert_frame_equal(top_n, item_exposure_lists())

    def test_repeated_calls_give_the_same_value(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        self.assertEqual(
            KullbackLeibler().evaluate(top_n, item_target()),
            KullbackLeibler().evaluate(top_n, item_target()),
        )

    def test_does_not_mutate_the_lists_when_using_relevance(self) -> None:
        """#75"""
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = binary_relevance().merge(user_groups(), on="user")
        KullbackLeibler().evaluate(top_n, user_target(), rel_matrix)
        assert_frame_equal(top_n, top_6().merge(user_groups(), on="user"))

    def test_does_not_mutate_the_relevance_matrix(self) -> None:
        """#75"""
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = binary_relevance().merge(user_groups(), on="user")
        KullbackLeibler().evaluate(top_n, user_target(), rel_matrix)
        assert_frame_equal(
            rel_matrix, binary_relevance().merge(user_groups(), on="user")
        )

    def test_repeated_calls_with_relevance_give_the_same_value(self) -> None:
        """#75"""
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = binary_relevance().merge(user_groups(), on="user")
        self.assertEqual(
            KullbackLeibler().evaluate(top_n, user_target(), rel_matrix),
            KullbackLeibler().evaluate(top_n, user_target(), rel_matrix),
        )


class MutualInformationTest(unittest.TestCase):
    def test_mutual_information_of_user_group_exposure(self) -> None:
        mi = MutualInformation().evaluate(user_exposure_lists(), "user")
        self.assertAlmostEqual(mi, 0.25582, delta=1e-5)

    def test_mutual_information_of_item_group_exposure(self) -> None:
        mi = MutualInformation().evaluate(item_exposure_lists(), "item")
        self.assertAlmostEqual(mi, 0.10570, delta=1e-5)

    def test_pairs_with_zero_probability_contribute_zero(self) -> None:
        """Pairs with zero joint probability are skipped (0 · log 0 = 0), so the result is finite."""
        mi = MutualInformation().evaluate(
            item_exposure_lists(), "item", relevant_except_group_3()
        )
        self.assertTrue(np.isfinite(mi))

    def test_raises_for_an_invalid_flag(self) -> None:
        with self.assertRaises(KeyError):
            MutualInformation().evaluate(item_exposure_lists(), "ratings")

    def test_invalid_flag_error_names_the_flag(self) -> None:
        with self.assertRaises(KeyError) as context:
            MutualInformation().evaluate(item_exposure_lists(), "ratings")

        self.assertIn("Invalid flag 'ratings'", str(context.exception))

    def test_raises_when_no_relevant_item_is_recommended(self) -> None:
        """With effectiveness, no relevant item recommended gives zero total exposure (#70)."""
        with self.assertRaises(DivisionByZeroException):
            MutualInformation().evaluate(*self._nothing_relevant_recommended())

    def test_division_error_names_the_metric(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            MutualInformation().evaluate(*self._nothing_relevant_recommended())

        self.assertEqual(str(context.exception), "MutualInformation: division by zero.")

    def test_does_not_mutate_the_input(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        MutualInformation().evaluate(top_n, "item")
        assert_frame_equal(top_n, item_exposure_lists())

    def test_repeated_calls_give_the_same_value(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        self.assertEqual(
            MutualInformation().evaluate(top_n, "item"),
            MutualInformation().evaluate(top_n, "item"),
        )

    @staticmethod
    def _nothing_relevant_recommended() -> tuple[pd.DataFrame, str, pd.DataFrame]:
        """Arguments of evaluate where no relevant item appears in the recommendations."""
        top_n = top_6().merge(user_groups(), on="user")
        rel_matrix = (
            binary_relevance()
            .assign(item="not recommended")
            .merge(user_groups(), on="user")
        )
        return top_n, "user", rel_matrix


class ExpMatrixTest(unittest.TestCase):
    def test_rank_one_gives_exposure_one_with_user_groups(self) -> None:
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

    def test_rank_one_gives_exposure_one_with_item_groups(self) -> None:
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

    def test_does_not_mutate_the_input(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        exp_matrix(top_n)
        assert_frame_equal(top_n, item_exposure_lists())


class ProbMatrixTest(unittest.TestCase):
    def test_equal_exposures_give_uniform_probabilities_with_user_groups(self) -> None:
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

    def test_equal_exposures_give_uniform_probabilities_with_item_groups(self) -> None:
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

    def test_does_not_mutate_the_input(self) -> None:
        """#75"""
        top_n = item_exposure_lists()
        prob_matrix(top_n)
        assert_frame_equal(top_n, item_exposure_lists())

    def test_does_not_mutate_a_float_rank_column(self) -> None:
        """A float rank column keeps its dtype, so it must not be overwritten in place (#75)."""
        top_n = item_exposure_lists().astype({"rank": float})
        prob_matrix(top_n)
        assert_frame_equal(top_n, item_exposure_lists().astype({"rank": float}))


class EffMatrixTest(unittest.TestCase):
    def test_effectiveness_with_binary_relevance(self) -> None:
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

    def test_effectiveness_with_graded_relevance(self) -> None:
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
