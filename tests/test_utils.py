import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from recsyslearn.errors.errors import ColumnsNotExistException
from recsyslearn.fairness.utils import eff_matrix, exp_matrix, prob_matrix
from recsyslearn.utils import check_columns_exist
from tests.utils import (
    first_example,
    item_groups,
    rel_matrix_1,
    rel_matrix_3,
    second_example,
    user_groups,
)


class ExpMatrixTest(unittest.TestCase):

    """
    Tester for the exp_matrix function.
    """

    def test_exp_matrix_1(self) -> None:
        top_n = first_example.merge(user_groups, on=["user"])
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
        top_n = first_example.merge(item_groups, on=["item"])
        result = exp_matrix(top_n)
        assert_frame_equal(
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
        top_n = first_example.merge(user_groups, on=["user"])
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
        top_n = first_example.merge(item_groups, on=["item"])
        result = prob_matrix(top_n)
        assert_frame_equal(
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
        top_n = second_example.merge(user_groups, on=["user"])
        rel_matrix = rel_matrix_1.merge(user_groups, on=["user"])
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
        top_n = second_example.merge(item_groups, on=["item"])
        rel_matrix = rel_matrix_3.merge(item_groups, on=["item"])
        result = eff_matrix(top_n, rel_matrix)
        result = result[result["rank"] != 0].reset_index(drop=True)
        assert_frame_equal(
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


class SmallUtilsTest(unittest.TestCase):
    def test_columns_exist_one(self):
        check_columns_exist(pd.DataFrame(columns=["user", "item"]), ["user"])

    def test_columns_dont_exist(self):
        with self.assertRaises(ColumnsNotExistException) as context:
            check_columns_exist(pd.DataFrame(columns=["user"]), ["user", "item"])

        self.assertTrue(
            "Dataframe does not contain columns. ['user', 'item']"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
