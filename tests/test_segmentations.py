import unittest
from recsyslearn.segmentations import ActivitySegmentation, PopularityPercentage, InteractionSegmentation
from tests.utils import dataset_item_example, dataset_user_example, dataset_popularity
from pandas.testing import assert_frame_equal


class InteractionSegmentationTest(unittest.TestCase):
    """
    Tester for the InteractionSegmentation class.
    """

    def setUp(self) -> None:
        self.interaction_segmenter = InteractionSegmentation()

    def test_segmentation(self) -> None:
        segmented_groups = self.interaction_segmenter.segment(dataset_item_example)
        self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '1', 'group'].eq('1').all())
        self.assertTrue(segmented_groups.loc[segmented_groups['item'] == '2', 'group'].eq('2').all())

    def test_segmentation_mantissa(self) -> None:
        self.interaction_segmenter.segment(dataset_item_example, [0.6, 0.3, 0.1])

    def test_segmentation_entire_dataset(self) -> None:
        self.assertIsNone(
            assert_frame_equal(
                dataset_item_example, self.interaction_segmenter.segment(dataset_item_example, [1])
            )
        )

    @unittest.expectedFailure
    def test_segmentation_not_supported(self) -> None:
        self.interaction_segmenter.segment(dataset_item_example, [0.7, 0.1, 0.1, 0.1])

    @unittest.expectedFailure
    def test_segmentation_wrong_proportion(self) -> None:
        self.interaction_segmenter.segment(dataset_item_example, [0.7, 0.4])


class ItemPopularityPercentageTest(unittest.TestCase):
    """
    Test for the PopularityPercentage class
    """

    def setUp(self) -> None:
        self.item_popularity_segmenter = PopularityPercentage()

    def test_popularity(self) -> None:
        popularity_dataframe = self.item_popularity_segmenter.segment(dataset_popularity)
        self.assertTrue((popularity_dataframe['percentage'] < 1.).all())
        self.assertAlmostEqual(1., popularity_dataframe['percentage'].sum(), delta=1e-5)
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['item'] == '1', 'percentage'].eq(0.6).all())
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['item'] == '2', 'percentage'].eq(0.3).all())
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['item'] == '3', 'percentage'].eq(0.1).all())


class UserPopularityPercentageTest(unittest.TestCase):
    """
    Test for the PopularityPercentage class
    """

    def setUp(self) -> None:
        self.user_popularity_segmenter = PopularityPercentage()

    def test_popularity(self) -> None:
        popularity_dataframe = self.user_popularity_segmenter.segment(dataset_popularity, group='user')
        self.assertTrue((popularity_dataframe['percentage'] < 1.).all())
        self.assertAlmostEqual(1., popularity_dataframe['percentage'].sum(), delta=1e-5)
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['user'] == '1', 'percentage'].eq(0.7).all())
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['user'] == '2', 'percentage'].eq(0.2).all())
        self.assertTrue(popularity_dataframe.loc[popularity_dataframe['user'] == '3', 'percentage'].eq(0.1).all())


class ActivitySegmentationTest(unittest.TestCase):
    """
    Test for the ActivitySegmentation class.
    """

    def setUp(self) -> None:
        self.user_segmenter = ActivitySegmentation()

    def test_segmentation(self) -> None:
        segmented_groups = self.user_segmenter.segment(dataset_user_example)
        self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '1', 'group'].eq('1').all())
        self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '2', 'group'].eq('2').all())
        self.assertTrue(segmented_groups.loc[segmented_groups['user'] == '3', 'group'].eq('2').all())

    def test_segmentation_mantissa(self) -> None:
        self.user_segmenter.segment(dataset_user_example, [0.6, 0.3, 0.1])

    def test_segmentation_entire_dataset(self) -> None:
        self.assertIsNone(
            assert_frame_equal(
                dataset_user_example, self.user_segmenter.segment(dataset_user_example, [1])
            )
        )

    @unittest.expectedFailure
    def test_segmentation_not_supported(self) -> None:
        self.user_segmenter.segment(dataset_user_example, [0.7, 0.1, 0.1, 0.1])

    @unittest.expectedFailure
    def test_segmentation_wrong_proportion(self) -> None:
        self.user_segmenter.segment(dataset_user_example, [0.7, 0.4])


if __name__ == '__main__':
    unittest.main()
