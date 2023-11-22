import unittest

from recsyslearn.errors.errors import (
    ColumnsNotExistException,
    InvalidValueException,
    RecListTooShortException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)


class TestErrors(unittest.TestCase):

    """
    Tester for the test_pattern function.
    """

    def test_exceptions(self) -> None:
        SegmentationNotSupportedException("")
        WrongProportionsException()
        RecListTooShortException(10)
        ColumnsNotExistException(["A", "B", "C"])
        InvalidValueException(-1)


if __name__ == "__main__":
    unittest.main()
