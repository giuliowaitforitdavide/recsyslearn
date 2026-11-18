import unittest

from recsyslearn.errors import base, metrics, segmentation, validation
from recsyslearn.errors.base import RecsyslearnException
from recsyslearn.errors.metrics import DivisionByZeroException, RecListTooShortException
from recsyslearn.errors.segmentation import (
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)
from recsyslearn.errors.validation import ColumnsNotExistException

ERROR_MODULES = (base, metrics, segmentation, validation)


class ErrorTest(unittest.TestCase):
    """
    Tester for the custom exceptions.
    """

    def test_exceptions(self) -> None:
        SegmentationNotSupportedException("")
        WrongProportionsException()
        RecListTooShortException(10)
        ColumnsNotExistException(["A", "B", "C"])
        InvalidValueException(-1)

    def test_common_base_class(self) -> None:
        """Every custom exception inherits from RecsyslearnException (#70)."""
        exceptions = [
            getattr(module, name) for module in ERROR_MODULES for name in module.__all__
        ]
        self.assertEqual(len(exceptions), 8)
        for exception in exceptions:
            with self.subTest(exception=exception.__name__):
                self.assertTrue(issubclass(exception, RecsyslearnException))

    def test_division_by_zero_exception(self) -> None:
        """DivisionByZeroException is also a ZeroDivisionError, so existing handlers keep working (#70)."""
        exception = DivisionByZeroException("NDCG", "user 5")
        self.assertIsInstance(exception, ZeroDivisionError)
        self.assertEqual(str(exception), "NDCG: division by zero for user 5.")
        self.assertEqual(
            str(DivisionByZeroException("Coverage")), "Coverage: division by zero."
        )


if __name__ == "__main__":
    unittest.main()
