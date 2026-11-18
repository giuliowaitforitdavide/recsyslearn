import unittest

import numpy as np

from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from recsyslearn.errors.metrics import DivisionByZeroException
from tests.fixtures import item_groups, item_popularity, top_5_unranked, top_6


class CoverageTest(unittest.TestCase):
    def test_coverage_empty_catalogue(self) -> None:
        """Coverage is undefined for an empty item catalogue (#70)."""
        with self.assertRaises(DivisionByZeroException) as context:
            Coverage().evaluate(top_5_unranked(), [])
        self.assertEqual(str(context.exception), "Coverage: division by zero.")

    def test_coverage_one(self) -> None:
        top_n = top_5_unranked()
        top_n = top_n[~top_n["item"].isin(["1", "3", "4"])]
        cov = Coverage().evaluate(top_n, item_groups().item.tolist())
        self.assertAlmostEqual(cov, 0.7)

    def test_coverage_two(self) -> None:
        top_n = top_5_unranked()
        top_n = top_n[~top_n["item"].isin(["1", "2", "4", "7", "9", "10"])]
        cov = Coverage().evaluate(top_n, item_groups().item.tolist())
        self.assertAlmostEqual(cov, 0.4)


class NoveltyTest(unittest.TestCase):
    def setUp(self):
        novelty = np.vectorize(lambda x: -np.log2(x))
        self.novelty = lambda list: float(np.mean(novelty(list)))

    def test_novelty_one(self) -> None:
        top_n = top_5_unranked().merge(item_groups(), on="item")
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self.novelty(top_n["group"].to_numpy()), delta=1e-5)

    def test_novelty_two(self) -> None:
        top_n = top_6().merge(item_groups(), on="item")
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self.novelty(top_n["group"].to_numpy()), delta=1e-5)

    def test_novelty_three(self) -> None:
        top_n = top_5_unranked().merge(item_popularity(), on="item")
        nov = Novelty().evaluate(top_n, popularity_definition="percentage")
        self.assertAlmostEqual(
            nov, self.novelty(top_n["percentage"].to_numpy()), delta=1e-5
        )


if __name__ == "__main__":
    unittest.main()
