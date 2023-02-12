import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from tests.utils import first_example, item_groups, item_pop_perc, second_example


class CoverageTest(unittest.TestCase):
    def test_coverage_one(self) -> None:
        top_n = first_example[
            (first_example["item"] != "3")
            & (first_example["item"] != "4")
            & (first_example["item"] != "1")
        ]
        cov = Coverage().evaluate(top_n, item_groups.item.tolist())
        self.assertAlmostEqual(cov, 0.7)

    def test_coverage_two(self) -> None:
        top_n = first_example[
            (first_example["item"] != "1")
            & (first_example["item"] != "2")
            & (first_example["item"] != "4")
            & (first_example["item"] != "7")
            & (first_example["item"] != "9")
            & (first_example["item"] != "10")
        ]
        cov = Coverage().evaluate(top_n, item_groups.item.tolist())
        self.assertAlmostEqual(cov, 0.4)


class NoveltyTest(unittest.TestCase):
    def setUp(self):
        self.novelty = lambda x: -np.log2(x)

    def test_novelty_one(self) -> None:
        top_n = first_example.merge(item_groups, on="item")
        nov = Novelty().evaluate(top_n)
        novelty_df = pd.DataFrame()
        novelty_df["user"] = top_n.user
        novelty_df["group"] = self.novelty(top_n.group)

        novelty_df = novelty_df.groupby("user").mean()
        novelty_df = novelty_df.group

        assert_series_equal(nov, novelty_df)

    def test_novelty_two(self) -> None:
        top_n = second_example.merge(item_groups, on="item")
        nov = Novelty().evaluate(top_n)

        novelty_df = pd.DataFrame()
        novelty_df["user"] = top_n.user
        novelty_df["group"] = self.novelty(top_n.group)

        novelty_df = novelty_df.groupby("user").mean()
        novelty_df = novelty_df.group

        assert_series_equal(nov, novelty_df)

    def test_novelty_three(self) -> None:
        top_n = first_example.merge(item_pop_perc, on="item")
        nov = Novelty().evaluate(top_n, popularity_definition="percentage")

        novelty_df = pd.DataFrame()
        novelty_df["user"] = top_n.user
        novelty_df["percentage"] = self.novelty(top_n.percentage)

        novelty_df = novelty_df.groupby("user").mean()
        novelty_df = novelty_df.percentage

        assert_series_equal(nov, novelty_df)


if __name__ == "__main__":
    unittest.main()
