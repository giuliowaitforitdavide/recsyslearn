import unittest
import numpy as np
import pandas as pd
from recsyslearn.dataset.utils import find_relevant_items
from pandas.testing import assert_frame_equal


class DatasetUtilsTest(unittest.TestCase):
    """
    Tester for the dataset utils.
    """

    def test_find_relevant_items(self) -> None:
        rel_matrix = pd.DataFrame([
            ['1', '2', 1],
            ['1', '9', 1],
            ['2', '6', 1],
            ['2', '7', 1],
            ['3', '1', 1],
            ['3', '3', 1],
            ['3', '4', 1],
            ['3', '5', 1],
            ['3', '7', 1],
            ['3', '9', 1],
            ['4', '2', 1],
            ['4', '3', 1],
            ['4', '5', 1],
            ['4', '7', 1],
            ['4', '8', 1],
            ['4', '9', 1],
            ['5', '1', 1],
            ['5', '2', 1],
            ['5', '4', 1],
            ['5', '5', 1],
            ['5', '6', 1],
            ['5', '9', 1],
            ['6', '9', 1],
        ], columns=['user', 'item', 'rank'])
        expected_result = pd.DataFrame([
            ['1', np.asarray(['2', '9'])],
            ['2', np.asarray(['6', '7'])],
            ['3', np.asarray(['1', '3', '4', '5', '7', '9'])],
            ['4', np.asarray(['2', '3', '5', '7', '8', '9'])],
            ['5', np.asarray(['1', '2', '4', '5', '6', '9'])],
            ['6', np.asarray(['9'])],
        ], columns=['user', 'pos_items'])

        pos_items = find_relevant_items(rel_matrix)
        assert_frame_equal(pos_items, expected_result)


if __name__ == '__main__':
    unittest.main()
