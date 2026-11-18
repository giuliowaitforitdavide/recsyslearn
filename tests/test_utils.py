"""Tests of the shared helpers in recsyslearn/utils.py."""

import unittest

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_series_equal

from recsyslearn.errors.metrics import DivisionByZeroException
from recsyslearn.errors.validation import ColumnsNotExistException
from recsyslearn.utils import check_columns_exist, safe_divide
from tests.case import TestCase


COLUMN_NAMES = ["user", "item", "rank", "group", "pos_items", "target_representation"]
NONZERO_FLOATS = st.floats(min_value=0.1, max_value=100, allow_nan=False).flatmap(
    lambda value: st.sampled_from([value, -value])
)


class CheckColumnsExistTest(TestCase):
    def test_accepts_a_dataframe_with_the_required_columns(self) -> None:
        check_columns_exist(pd.DataFrame(columns=["user", "item"]), ["user"])

    def test_raises_when_a_column_is_missing(self) -> None:
        with self.assertRaisesMessage(
            ColumnsNotExistException,
            "Dataframe does not contain columns. ['user', 'item']",
        ):
            check_columns_exist(pd.DataFrame(columns=["user"]), ["user", "item"])

    @given(st.sets(st.sampled_from(COLUMN_NAMES), min_size=1), st.data())
    def test_accepts_any_subset_of_the_columns(self, present, data) -> None:
        required = data.draw(st.lists(st.sampled_from(sorted(present)), unique=True))
        check_columns_exist(pd.DataFrame(columns=sorted(present)), required)

    @given(st.sets(st.sampled_from(COLUMN_NAMES), max_size=5), st.data())
    def test_raises_when_any_required_column_is_missing(self, present, data) -> None:
        missing = data.draw(st.sampled_from(sorted(set(COLUMN_NAMES) - present)))
        required = sorted(present) + [missing]
        with self.assertRaisesMessage(
            ColumnsNotExistException, f"Dataframe does not contain columns. {required}"
        ):
            check_columns_exist(pd.DataFrame(columns=sorted(present)), required)


class SafeDivideTest(TestCase):
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
        with self.assertRaisesMessage(
            DivisionByZeroException, "NDCG: division by zero."
        ):
            safe_divide(1, 0, metric="NDCG")

    def test_raises_on_any_zero_in_an_array_denominator(self) -> None:
        with self.assertRaisesMessage(DivisionByZeroException, "M: division by zero."):
            safe_divide(np.ones(3), np.array([1.0, 0.0, 2.0]), metric="M")

    def test_error_names_the_given_label(self) -> None:
        with self.assertRaisesMessage(
            DivisionByZeroException, "NDCG: division by zero for user 5."
        ):
            safe_divide(1, 0, metric="NDCG", labels="user 5")

    def test_error_names_the_labels_of_the_zero_denominators(self) -> None:
        labels = pd.Series(["a", "b", "c"], name="group")
        with self.assertRaisesMessage(
            DivisionByZeroException,
            "KullbackLeibler: division by zero for group b, c.",
        ):
            safe_divide(
                pd.Series([1.0, 1.0, 1.0]),
                pd.Series([1.0, 0.0, 0.0]),
                metric="KullbackLeibler",
                labels=labels,
            )

    def test_error_lists_the_labels_in_sorted_order(self) -> None:
        """#84: the message must not depend on the order of the input rows."""
        with self.assertRaisesMessage(
            DivisionByZeroException, "M: division by zero for user 1, 2, 3."
        ):
            safe_divide(
                np.ones(3),
                np.zeros(3),
                metric="M",
                labels=pd.Series(["3", "1", "2"], name="user"),
            )

    def test_error_sorts_labels_of_mixed_types_by_their_text(self) -> None:
        with self.assertRaisesMessage(
            DivisionByZeroException, "M: division by zero for group 1, a."
        ):
            safe_divide(
                np.ones(2),
                np.zeros(2),
                metric="M",
                labels=pd.Series(["a", 1], name="group", dtype=object),
            )

    @given(st.permutations([f"u{index}" for index in range(1, 8)]))
    def test_error_message_does_not_depend_on_the_label_order(self, users) -> None:
        with self.assertRaisesMessage(
            DivisionByZeroException,
            "M: division by zero for user u1, u2, u3, u4, u5 and 2 more.",
        ):
            safe_divide(
                np.ones(len(users)),
                np.zeros(len(users)),
                metric="M",
                labels=pd.Series(users, name="user"),
            )

    @given(st.lists(st.tuples(NONZERO_FLOATS, NONZERO_FLOATS), min_size=1, max_size=20))
    def test_equals_true_division_when_no_denominator_is_zero(self, pairs) -> None:
        numerator, denominator = (np.array(values) for values in zip(*pairs))
        np.testing.assert_array_equal(
            safe_divide(numerator, denominator, metric="M"), numerator / denominator
        )

    @given(st.lists(NONZERO_FLOATS, min_size=1, max_size=20), st.data())
    def test_raises_when_any_denominator_is_zero(self, denominator, data) -> None:
        zero_at = data.draw(st.integers(min_value=0, max_value=len(denominator) - 1))
        denominator[zero_at] = 0.0
        with self.assertRaisesMessage(DivisionByZeroException, "M: division by zero."):
            safe_divide(np.ones(len(denominator)), np.array(denominator), metric="M")


if __name__ == "__main__":
    unittest.main()
