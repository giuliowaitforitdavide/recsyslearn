import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty

item_pop_perc = pd.DataFrame([
    ['1', 0.05],
    ['2', 0.01],
    ['3', 0.03],
    ['4', 0.23],
    ['5', 0.4],
    ['6', 0.6],
    ['7', 0.15],
    ['8', 0.34],
    ['9', 0.07],
    ['10', 0.02],
], columns=['item', 'percentage'])

item_groups = pd.DataFrame([
    ['1', '1'],
    ['2', '1'],
    ['3', '2'],
    ['4', '2'],
    ['5', '2'],
    ['6', '3'],
    ['7', '3'],
    ['8', '3'],
    ['9', '3'],
    ['10', '3']
], columns=['item', 'group'])

first_example = pd.DataFrame([
    ['1', '3', 1],
    ['1', '4', 1],
    ['1', '6', 1],
    ['1', '7', 1],
    ['1', '8', 1],
    ['2', '1', 1],
    ['2', '2', 1],
    ['2', '5', 1],
    ['2', '6', 1],
    ['2', '7', 1],
    ['3', '2', 1],
    ['3', '3', 1],
    ['3', '6', 1],
    ['3', '9', 1],
    ['3', '10', 1],
    ['4', '1', 1],
    ['4', '3', 1],
    ['4', '6', 1],
    ['4', '7', 1],
    ['4', '9', 1],
    ['5', '1', 1],
    ['5', '3', 1],
    ['5', '5', 1],
    ['5', '7', 1],
    ['5', '9', 1],
    ['6', '2', 1],
    ['6', '3', 1],
    ['6', '5', 1],
    ['6', '9', 1],
    ['6', '10', 1],
], columns=['user', 'item', 'rank'])

second_example = pd.DataFrame([
    ['1', '1', 1],
    ['1', '3', 6],
    ['1', '4', 3],
    ['1', '5', 4],
    ['1', '8', 5],
    ['1', '9', 2],
    ['2', '2', 6],
    ['2', '3', 5],
    ['2', '6', 4],
    ['2', '7', 3],
    ['2', '8', 2],
    ['2', '9', 1],
    ['3', '2', 1],
    ['3', '3', 2],
    ['3', '4', 3],
    ['3', '7', 4],
    ['3', '8', 5],
    ['3', '9', 6],
    ['4', '1', 4],
    ['4', '3', 5],
    ['4', '4', 6],
    ['4', '7', 3],
    ['4', '8', 2],
    ['4', '9', 1],
    ['5', '1', 6],
    ['5', '3', 2],
    ['5', '5', 3],
    ['5', '6', 4],
    ['5', '8', 5],
    ['5', '9', 1],
    ['6', '2', 3],
    ['6', '3', 5],
    ['6', '5', 4],
    ['6', '6', 6],
    ['6', '8', 2],
    ['6', '9', 1]
], columns=['user', 'item', 'rank'])

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

    def test_novelty_one(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        novelty_df = Novelty().evaluate(top_n)
        expected_novelty_df = pd.DataFrame([
            ['1', -1.350978],
            ['2', -0.833985],
            ['3', -1.150978],
            ['4', -1.150978],
            ['5', -1.033985],
            ['6', -1.033985],
        ], columns=['user', 'rank'])

        assert_frame_equal(novelty_df, expected_novelty_df)

    def test_novelty_two(self) -> None:
        top_n = second_example.merge(item_groups, on='item')
        novelty_df = Novelty().evaluate(top_n)

        expected_novelty_df = pd.DataFrame([
            ['1', -1.028321],
            ['2', -1.223308],
            ['3', -1.125815],
            ['4', -1.125815],
            ['5', -1.125815],
            ['6', -1.125815],
        ], columns=['user', 'rank'])

        assert_frame_equal(novelty_df, expected_novelty_df)


    def test_novelty_three(self) -> None:
        top_n = first_example.merge(item_pop_perc, on='item')
        novelty_df = Novelty().evaluate(top_n, popularity_definition='percentage')

        expected_novelty_df = pd.DataFrame([
            ['1', 2.441902],
            ['2', 3.152329],
            ['3', 4.384015],
            ['4', 3.338251],
            ['5', 3.455243],
            ['6', 4.501007],
        ], columns=['user', 'rank'])

        assert_frame_equal(novelty_df, expected_novelty_df)


if __name__ == "__main__":
    unittest.main()
