import unittest

import numpy as np
import pandas as pd

from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from recsyslearn.errors.metrics import DivisionByZeroException
from tests.fixtures import item_groups, item_popularity, top_5_unranked, top_6


class CoverageTest(unittest.TestCase):
    def test_seven_of_ten_items_recommended(self) -> None:
        top_n = top_5_unranked()
        top_n = top_n[~top_n["item"].isin(["1", "3", "4"])]
        cov = Coverage().evaluate(top_n, item_groups().item.tolist())
        self.assertAlmostEqual(cov, 0.7)

    def test_four_of_ten_items_recommended(self) -> None:
        top_n = top_5_unranked()
        top_n = top_n[~top_n["item"].isin(["1", "2", "4", "7", "9", "10"])]
        cov = Coverage().evaluate(top_n, item_groups().item.tolist())
        self.assertAlmostEqual(cov, 0.4)

    def test_raises_for_an_empty_catalogue(self) -> None:
        """Coverage is undefined for an empty item catalogue (#70)."""
        with self.assertRaises(DivisionByZeroException):
            Coverage().evaluate(top_5_unranked(), [])

    def test_division_error_names_the_metric(self) -> None:
        with self.assertRaises(DivisionByZeroException) as context:
            Coverage().evaluate(top_5_unranked(), [])

        self.assertEqual(str(context.exception), "Coverage: division by zero.")


class NoveltyTest(unittest.TestCase):
    def test_uses_the_group_column_as_popularity_by_default(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self._expected(top_n["group"]), delta=1e-5)

    def test_does_not_depend_on_the_rank(self) -> None:
        """top_6 has different ranks per item; the expected value ignores them."""
        top_n = top_6().merge(item_groups(), on="item")
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self._expected(top_n["group"]), delta=1e-5)

    def test_uses_the_given_popularity_column(self) -> None:
        top_n = top_5_unranked().merge(item_popularity(), on="item")
        nov = Novelty().evaluate(top_n, popularity_definition="percentage")
        self.assertAlmostEqual(nov, self._expected(top_n["percentage"]), delta=1e-5)

    @staticmethod
    def _expected(popularity: pd.Series) -> float:
        """Mean of -log2(popularity) over all recommended items."""
        return float(np.mean(-np.log2(popularity.astype(float).to_numpy())))


if __name__ == "__main__":
    unittest.main()
