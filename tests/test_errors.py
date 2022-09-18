import unittest
from recsyslearn.errors.errors import FlagNotValidException, SegmentationNotSupportedException, InvalidValueException, WrongProportionsException, RecListTooShortException, ColumnsNotExistException


class TestErrors(unittest.TestCase):

    """
    Tester for the test_pattern function.
    """

    def test_exceptions(self) -> None:
        FlagNotValidException()
        SegmentationNotSupportedException()
        WrongProportionsException()
        RecListTooShortException(10)
        ColumnsNotExistException(['A', 'B', 'C'])
        InvalidValueException(-1)


if __name__ == "__main__":
    unittest.main()
