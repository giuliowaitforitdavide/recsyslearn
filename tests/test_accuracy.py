import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.errors.errors import RecListTooShortException

import pandas as pd
import numpy as np

top_n = pd.DataFrame([
    ['1', '2', 1],
    ['1', '8', 2],
    ['1', '5', 3],
    ['1', '3', 4],
    ['1', '6', 5],
    ['2', '1', 1],
    ['2', '2', 2],
    ['2', '6', 3],
    ['2', '7', 4],
    ['2', '9', 5],
    ['3', '6', 1],
    ['3', '5', 2],
    ['3', '8', 3],
    ['3', '7', 4],
    ['3', '9', 5],
    ['4', '1', 1],
    ['4', '2', 2],
    ['4', '3', 3],
    ['4', '4', 4],
    ['4', '5', 5],
    ['5', '5', 1],
    ['5', '4', 2],
    ['5', '3', 3],
    ['5', '2', 4],
    ['5', '1', 5],
    ['6', '9', 1],
    ['6', '8', 2],
    ['6', '6', 3],
    ['6', '3', 4],
    ['6', '4', 5],
], columns=['user', 'item', 'rank'])

pos_items = pd.DataFrame([
    ['1', np.asarray(['2', '9'])],
    ['2', np.asarray(['6', '7'])],
    ['3', np.asarray(['1', '3', '4', '5', '7', '9'])],
    ['4', np.asarray(['2', '3', '5', '7', '8', '9'])],
    ['5', np.asarray(['1', '2', '4', '5', '6', '9'])],
    ['6', np.asarray(['9'])],
], columns=['user', 'pos_items'])


class NDCGTest(unittest.TestCase):

    def test_ndcg(self) -> None:
        ndcg_df = NDCG().evaluate(top_n, pos_items, ats=(2,))
        ndcg_vals = pd.DataFrame(
            [
                ['1', (1 / np.log(1 + 1) + 0 / np.log(2 + 1)) /
                 (1 / np.log(1 + 1) + 1 / np.log(2 + 1))],
                ['2', (0 / np.log(1 + 1) + 0 / np.log(2 + 1)) /
                 (1 / np.log(1 + 1) + 1 / np.log(2 + 1))],
                ['3', (0 / np.log(1 + 1) + 1 / np.log(2 + 1)) /
                 (1 / np.log(1 + 1) + 1 / np.log(2 + 1))],
                ['4', (0 / np.log(1 + 1) + 1 / np.log(2 + 1)) /
                 (1 / np.log(1 + 1) + 1 / np.log(2 + 1))],
                ['5', (1 / np.log(1 + 1) + 1 / np.log(2 + 1)) /
                 (1 / np.log(1 + 1) + 1 / np.log(2 + 1))],
                ['6', (1 / np.log(1 + 1) + 0 / np.log(2 + 1)) /
                 (1 / np.log(1 + 1) + 0 / np.log(2 + 1))]
            ], columns=['user', 'NDCG@2']
        )
        assert_frame_equal(ndcg_df, ndcg_vals)

    def test_ndcg_error(self) -> None:
        with self.assertRaises(RecListTooShortException):
            NDCG().evaluate(top_n.iloc[[0, 1]], pos_items, ats=(3,))


if __name__ == '__main__':
    unittest.main()
