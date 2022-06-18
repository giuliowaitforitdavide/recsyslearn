import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from recsyslearn.utils import test_pattern, exp_matrix
from tests.utils import first_example, user_groups

class TestPattern(unittest.TestCase):

    '''
    Tester for the test_pattern function.
    '''

    def test_pattern(self) -> None:
        test_pattern(pd.DataFrame(columns=['a', 'b', 'c']), ['a', 'b', 'c'])


    @unittest.expectedFailure
    def test_failure(self) -> None:
        test_pattern(['a', 'b', 'c'], ['a', 'b'])


class TestExpMatrix(unittest.TestCase):


    '''
    Tester for the exp_matrix function.
    '''
    
    def test_exp_matrix(self) -> None:
        top_n = first_example.merge(user_groups, on=['user'])
        result = exp_matrix(top_n)
        assert_frame_equal(result, pd.DataFrame([
			['1', '3', 1.0, '1'],
			['1', '4', 1.0, '1'],
			['1', '6', 1.0, '1'],
			['1', '7', 1.0, '1'],
			['1', '8', 1.0, '1'],
			['2', '1', 1.0, '1'],
			['2', '2', 1.0, '1'],
			['2', '5', 1.0, '1'],
			['2', '6', 1.0, '1'],
			['2', '7', 1.0, '1'],
			['3', '2', 1.0, '2'],
			['3', '3', 1.0, '2'],
			['3', '6', 1.0, '2'],
			['3', '9', 1.0, '2'],
			['3', '10', 1.0, '2'],
			['4', '1', 1.0, '2'],
			['4', '3', 1.0, '2'],
			['4', '6', 1.0, '2'],
			['4', '7', 1.0, '2'],
			['4', '9', 1.0, '2'],
			['5', '1', 1.0, '2'],
			['5', '3', 1.0, '2'],
			['5', '5', 1.0, '2'],
			['5', '7', 1.0, '2'],
			['5', '9', 1.0, '2'],
			['6', '2', 1.0, '2'],
			['6', '3', 1.0, '2'],
			['6', '5', 1.0, '2'],
			['6', '9', 1.0, '2'],
			['6', '10', 1.0, '2']
        ], columns=['user', 'item', 'rank', 'group']),
        check_like=True,
        check_exact=False)
