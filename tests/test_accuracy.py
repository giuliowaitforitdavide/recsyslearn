import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.errors.errors import RecListTooShortException
from tests.utils import pos_items, top_n_1


class NDCGTest(unittest.TestCase):

    def test_ndcg(self) -> None:
        ndcg_df = NDCG().evaluate(top_n_1, pos_items, ats=(2,))
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
        with self.assertRaises(RecListTooShortException) as context:
            NDCG().evaluate(top_n_1.iloc[[0, 1]], pos_items, ats=(3,))

        self.assertTrue("Values for the recommendation list length is not compatbile with the dataset. ats=(3,)" in str(
            context.exception))


if __name__ == '__main__':
    unittest.main()
