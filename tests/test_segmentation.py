import unittest
from recsyslearn.segmentations import ActivitySegmentation, PopularityPercentage, InteractionSegmentation
import pandas as pd
from tests.utils import dataset_item_example, dataset_user_example, dataset_popularity


class InteractionSegmentationTest(unittest.TestCase):
    """
    Tester for the InteractionSegmentation class.
    """

    def setUp(self) -> None:
        self.interactionSegmentator = InteractionSegmentation()

    def test_segmentation(self) -> None:
        segmented_groups = self.interactionSegmentator.segment(dataset_item_example)
        self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '1', 'group'].eq('1').all())
        self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '2', 'group'].eq('2').all())


class ItemPopularityPercentageTest(unittest.TestCase):
    """
    Test for the PopularityPercentage class
    """

    def setUp(self) -> None:
        self.interactionSegmentator = PopularityPercentage()

    def test_popularity(self) -> None:
        popularity_dataframe = self.interactionSegmentator.segment(dataset_popularity)
        # Every percentage should be less than 1
        self.assertTrue((popularity_dataframe['percentage'] < 1.).all())
        # The sum of all percentages should be (roughly) 1
        self.assertTrue(1. - popularity_dataframe['percentage'].sum() < 0.001)
        # Check the individual item percentages
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['item'] == '1', 'percentage'].eq(0.6).all())
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['item'] == '2', 'percentage'].eq(0.3).all())
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['item'] == '3', 'percentage'].eq(0.1).all())


class ActivitySegmentationTest(unittest.TestCase):
    """
    Test for the ActivitySegmentation class.
    """

    def setUp(self) -> None:
        self.userSegmentator = ActivitySegmentation()

    def test_segmentation(self) -> None:
        segmented_groups = self.userSegmentator.segment(dataset_user_example)
        self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '1', 'group'].eq('1').all())
        self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '2', 'group'].eq('2').all())
        self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '3', 'group'].eq('2').all())


if __name__ == '__main__':
    unittest.main()
