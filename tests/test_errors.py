import unittest
from recsyslearn.errors import ColumnsNotMatchException, FlagNotValidException, SegmentationNotSupportedException, WrongProportionsException


class TestErrors(unittest.TestCase):

    """
    Tester for the test_pattern function.
    """

    def test_exceptions(self) -> None:
        columns_exception = ColumnsNotMatchException(['A', 'B', 'C'])
        flag_not_valid_exception = FlagNotValidException()
        segmentation_not_supported_exception = SegmentationNotSupportedException()
        wrong_proportion_exception = WrongProportionsException()


if __name__ == "__main__":
    unittest.main()
