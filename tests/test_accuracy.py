import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.errors.metrics import DivisionByZeroException, RecListTooShortException
from tests.fixtures import pos_items, top_5


class NDCGTest(unittest.TestCase):
    def test_matches_the_formula_at_cutoff_2(self) -> None:
        ndcg_df = NDCG().evaluate(top_5(), pos_items(), ats=(2,))
        ndcg_vals = pd.DataFrame(
            [
                [
                    "1",
                    (1 / np.log(1 + 1) + 0 / np.log(2 + 1))
                    / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                ],
                [
                    "2",
                    (0 / np.log(1 + 1) + 0 / np.log(2 + 1))
                    / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                ],
                [
                    "3",
                    (0 / np.log(1 + 1) + 1 / np.log(2 + 1))
                    / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                ],
                [
                    "4",
                    (0 / np.log(1 + 1) + 1 / np.log(2 + 1))
                    / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                ],
                [
                    "5",
                    (1 / np.log(1 + 1) + 1 / np.log(2 + 1))
                    / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                ],
                [
                    "6",
                    (1 / np.log(1 + 1) + 0 / np.log(2 + 1))
                    / (1 / np.log(1 + 1) + 0 / np.log(2 + 1)),
                ],
            ],
            columns=["user", "NDCG@2"],
        )
        assert_frame_equal(ndcg_df, ndcg_vals)

    def test_warns_about_cutoffs_longer_than_the_shortest_list(self) -> None:
        """The warning lists the skipped cut-offs (#69)."""
        with self.assertWarnsRegex(UserWarning, r"\[10, 20\]"):
            NDCG().evaluate(top_5(), pos_items(), ats=(2, 10, 20))

    def test_computes_only_the_cutoffs_within_the_shortest_list(self) -> None:
        """Cut-offs longer than the shortest list are skipped, the others are computed (#69)."""
        with self.assertWarns(UserWarning):
            ndcg_df = NDCG().evaluate(top_5(), pos_items(), ats=(2, 10, 20))

        assert_frame_equal(ndcg_df, NDCG().evaluate(top_5(), pos_items(), ats=(2,)))

    def test_raises_when_no_cutoff_is_computable(self) -> None:
        with self.assertRaises(RecListTooShortException):
            NDCG().evaluate(top_5().iloc[[0, 1]], pos_items(), ats=(3,))

    def test_too_short_error_lists_the_requested_cutoffs(self) -> None:
        with self.assertRaises(RecListTooShortException) as context:
            NDCG().evaluate(top_5().iloc[[0, 1]], pos_items(), ats=(3,))

        self.assertIn("ats=(3,)", str(context.exception))

    def test_raises_for_user_without_relevant_items(self) -> None:
        """The ideal DCG of a user without relevant items is zero: NDCG is undefined (#70)."""
        with self.assertRaises(DivisionByZeroException):
            NDCG().evaluate(self._user_1_lists(), self._no_relevant_items(), ats=(2,))

    def test_division_error_names_the_user(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            NDCG().evaluate(self._user_1_lists(), self._no_relevant_items(), ats=(2,))

        self.assertEqual(str(context.exception), "NDCG: division by zero for user 1.")

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
