import unittest
from recsyslearn.segmentations import ActivitySegmentation, InteractionSegmentation
import pandas as pd

interactionSegmentator = InteractionSegmentation()

userSegmentator = ActivitySegmentation()

dataset_item_example = pd.DataFrame([
  ['1', '1', 1],
  ['2', '1', 1],
  ['3', '1', 1],
  ['1', '2', 1],
  ['1', '2', 1]
], columns=['user', 'item', 'rank'])

dataset_user_example = pd.DataFrame([
  ['1', '1', 1],
  ['1', '2', 1],
  ['1', '3', 1],
  ['1', '4', 1],
  ['1', '5', 1],
  ['1', '6', 1],
  ['1', '7', 1],
  ['1', '8', 1],
  ['1', '9', 1],
  ['1', '1', 1],
  ['1', '2', 1],
  ['1', '3', 1],
  ['1', '4', 1],
  ['1', '5', 1],
  ['1', '6', 1],
  ['1', '7', 1],
  ['1', '8', 1],
  ['1', '9', 1], 
  ['1', '1', 1],
  ['1', '2', 1],
  ['1', '3', 1],
  ['1', '4', 1],
  ['1', '5', 1],
  ['1', '6', 1],
  ['1', '7', 1],
  ['1', '8', 1],
  ['1', '9', 1], 
  ['1', '1', 1],
  ['1', '2', 1],
  ['1', '3', 1],
  ['1', '4', 1],
  ['1', '5', 1],
  ['1', '6', 1],
  ['1', '7', 1],
  ['1', '8', 1],
  ['1', '9', 1], 
  ['2', '1', 1],
  ['2', '1', 1],
  ['3', '1', 1]
], columns=['user', 'item', 'rank'])


class InteractionSegmentationTest(unittest.TestCase):

  def test_segmentation(self) -> None:
    segmented_groups = interactionSegmentator.segment(dataset_item_example)
    self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '1', 'group'].eq('1').all())
    self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '2', 'group'].eq('2').all())


class ActivitySegmentationTest(unittest.TestCase):

  def test_segmentation(self) -> None:
    segmented_groups = userSegmentator.segment(dataset_user_example)
    self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '1', 'group'].eq('1').all())
    self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '2', 'group'].eq('2').all())
    self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '3', 'group'].eq('2').all())


if __name__ == '__main__':
  unittest.main()
