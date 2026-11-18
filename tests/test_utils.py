"""Tests of the shared helpers in recsyslearn/utils.py."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from recsyslearn.errors.metrics import DivisionByZeroException
from recsyslearn.errors.validation import ColumnsNotExistException
from recsyslearn.utils import check_columns_exist, safe_divide


class CheckColumnsExistTest(unittest.TestCase):
    """check_columns_exist accepts DataFrames with the required columns and rejects the others."""

    def test_columns_exist_one(self):
        check_columns_exist(pd.DataFrame(columns=["user", "item"]), ["user"])

    def test_columns_dont_exist(self):
        with self.assertRaises(ColumnsNotExistException) as context:
            check_columns_exist(pd.DataFrame(columns=["user"]), ["user", "item"])

        self.assertTrue(
            "Dataframe does not contain columns. ['user', 'item']"
            in str(context.exception)
        )


class SafeDivideTest(unittest.TestCase):
    """safe_divide divides like `/` and raises DivisionByZeroException on any zero denominator (#70)."""

    def test_scalar(self) -> None:
        self.assertEqual(safe_divide(1, 4, metric="M"), 0.25)

    def test_series(self) -> None:
        numerator = pd.Series([1.0, 2.0, 3.0])
        denominator = pd.Series([2.0, 4.0, 6.0])
        assert_series_equal(
            safe_divide(numerator, denominator, metric="M"), numerator / denominator
        )

    def test_scalar_zero(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            safe_divide(1, 0, metric="NDCG", labels="user 5")
        self.assertEqual(str(context.exception), "NDCG: division by zero for user 5.")

    def test_series_zero_names_the_labels(self) -> None:
        labels = pd.Series(["a", "b", "c"], name="group")
        with self.assertRaises(DivisionByZeroException) as context:
            safe_divide(
                pd.Series([1.0, 1.0, 1.0]),
                pd.Series([1.0, 0.0, 0.0]),
                metric="KullbackLeibler",
                labels=labels,
            )
        self.assertEqual(
            str(context.exception),
            "KullbackLeibler: division by zero for group b, c.",
        )

    def test_numpy_zero_is_a_zero_division_error(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            safe_divide(np.ones(3), np.array([1.0, 0.0, 2.0]), metric="M")


if __name__ == "__main__":
    unittest.main()
