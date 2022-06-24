import unittest
from recsyslearn.segmentations import ActivitySegmentation, InteractionSegmentation
import pandas as pd
from tests.utils import dataset_item_example, dataset_user_example

class InteractionSegmentationTest(unittest.TestCase):

	'''
	Tester for the InteractionSegmentation class.
	'''

	def setUp(self) -> None:
		self.interactionSegmentator = InteractionSegmentation()

	def test_segmentation(self) -> None:
		segmented_groups = self.interactionSegmentator.segment(dataset_item_example)
		self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '1', 'group'].eq('1').all())
		self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '2', 'group'].eq('2').all())


class ActivitySegmentationTest(unittest.TestCase):

	'''
	Test for the ActivitySegmentation class.
	'''

	def setUp(self) -> None:
		self.userSegmentator = ActivitySegmentation()

	def test_segmentation(self) -> None:
		segmented_groups = self.userSegmentator.segment(dataset_user_example)
		self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '1', 'group'].eq('1').all())
		self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '2', 'group'].eq('2').all())
		self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '3', 'group'].eq('2').all())


if __name__ == '__main__':
	unittest.main()
