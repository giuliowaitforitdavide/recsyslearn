import unittest
import pandas as pd

from recsyslearn.utils import test_pattern

class TestPattern(unittest.TestCase):

  def test_pattern(self) -> None:
    test_pattern(pd.DataFrame(columns=['a', 'b', 'c']), ['a', 'b', 'c'])
    self.assertTrue(True)


  @unittest.expectedFailure
  def test_failure(self) -> None:
    test_pattern(['a', 'b', 'c'], ['a', 'b'])
