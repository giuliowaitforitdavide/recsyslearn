import unittest
from recsyslearn.errors import ColumnsNotMatchException, \
    FlagNotValidException, \
    SegmentationNotSupportedException, \
    WrongProportionsException, \
    InvalidValueException


class TestErrors(unittest.TestCase):

    """
    Tester for the test_pattern function.
    """

    def test_exceptions(self) -> None:
        _columns_exception = ColumnsNotMatchException(['A', 'B', 'C'])
        _flag_not_valid_exception = FlagNotValidException()
        _segmentation_not_supported_exception = SegmentationNotSupportedException()
        _wrong_proportion_exception = WrongProportionsException()
        _invalid_value_exception = InvalidValueException(-1)


if __name__ == "__main__":
    unittest.main()
