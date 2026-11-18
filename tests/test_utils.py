"""Tests of the shared helpers in recsyslearn/utils.py."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from recsyslearn.errors.metrics import DivisionByZeroException
from recsyslearn.errors.validation import ColumnsNotExistException
from recsyslearn.utils import check_columns_exist, safe_divide


class CheckColumnsExistTest(unittest.TestCase):
    def test_accepts_a_dataframe_with_the_required_columns(self) -> None:
        check_columns_exist(pd.DataFrame(columns=["user", "item"]), ["user"])

    def test_raises_when_a_column_is_missing(self) -> None:
        with self.assertRaises(ColumnsNotExistException):
            check_columns_exist(pd.DataFrame(columns=["user"]), ["user", "item"])

    def test_error_lists_the_required_columns(self) -> None:
        with self.assertRaises(ColumnsNotExistException) as context:
            check_columns_exist(pd.DataFrame(columns=["user"]), ["user", "item"])

        self.assertEqual(
            str(context.exception),
            "Dataframe does not contain columns. ['user', 'item']",
        )


class SafeDivideTest(unittest.TestCase):
    """safe_divide divides like `/` and raises DivisionByZeroException on any zero denominator (#70)."""

    def test_divides_scalars(self) -> None:
        self.assertEqual(safe_divide(1, 4, metric="M"), 0.25)

    def test_divides_series_element_wise(self) -> None:
        numerator = pd.Series([1.0, 2.0, 3.0])
        denominator = pd.Series([2.0, 4.0, 6.0])
        assert_series_equal(
            safe_divide(numerator, denominator, metric="M"), numerator / denominator
        )

    def test_raises_on_a_zero_scalar_denominator(self) -> None:
        with self.assertRaises(DivisionByZeroException):
            safe_divide(1, 0, metric="NDCG")

    def test_raises_on_any_zero_in_an_array_denominator(self) -> None:
        with self.assertRaises(DivisionByZeroException):
            safe_divide(np.ones(3), np.array([1.0, 0.0, 2.0]), metric="M")

    def test_error_names_the_given_label(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            safe_divide(1, 0, metric="NDCG", labels="user 5")

        self.assertEqual(str(context.exception), "NDCG: division by zero for user 5.")

    def test_error_names_the_labels_of_the_zero_denominators(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
