import unittest
from recsyslearn.errors import ColumnsNotMatchException, \
    FlagNotValidException, \
    SegmentationNotSupportedException, \
    InvalidValueException, \
    WrongProportionsException, \
    ListTooShortException, \
    ColumnsNotExistException


class TestErrors(unittest.TestCase):

    """
    Tester for the test_pattern function.
    """

    def test_exceptions(self) -> None:
        _columns_exception = ColumnsNotMatchException(['A', 'B', 'C'])
        _flag_not_valid_exception = FlagNotValidException()
        _segmentation_not_supported_exception = SegmentationNotSupportedException()
        _wrong_proportion_exception = WrongProportionsException()
        _list_too_short_exception = ListTooShortException(10)
        _columns_not_exist_exception = ColumnsNotExistException(['A', 'B', 'C'])
        _invalid_value_exception = InvalidValueException(-1)


if __name__ == "__main__":
    unittest.main()
