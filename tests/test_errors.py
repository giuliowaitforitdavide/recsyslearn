import unittest

from recsyslearn.errors import base, metrics, segmentation, validation
from recsyslearn.errors.base import RecsyslearnException
from recsyslearn.errors.metrics import DivisionByZeroException, RecListTooShortException
from recsyslearn.errors.segmentation import (
    InvalidGroupException,
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)
from recsyslearn.errors.validation import ColumnsNotExistException
from tests.case import TestCase

ERROR_MODULES = (base, metrics, segmentation, validation)


class ExceptionHierarchyTest(TestCase):
    def test_error_modules_export_all_eight_exceptions(self) -> None:
        self.assertEqual(len(self._exported_exceptions()), 8)

    def test_every_exception_inherits_from_the_base_class(self) -> None:
        """Every custom exception inherits from RecsyslearnException (#70)."""
        for exception in self._exported_exceptions():
            with self.subTest(exception=exception.__name__):
                self.assertTrue(issubclass(exception, RecsyslearnException))

    def test_division_by_zero_is_a_zero_division_error(self) -> None:
        """Existing `except ZeroDivisionError` handlers keep working (#70)."""
        self.assertTrue(issubclass(DivisionByZeroException, ZeroDivisionError))

    @staticmethod
    def _exported_exceptions() -> list[type]:
        """The exceptions listed in __all__ of every error module."""
        return [
            getattr(module, name) for module in ERROR_MODULES for name in module.__all__
        ]


class ExceptionMessageTest(TestCase):
    def test_division_by_zero_names_the_metric_and_the_location(self) -> None:
        self.assertEqual(
            str(DivisionByZeroException("NDCG", "user 5")),
            "NDCG: division by zero for user 5.",
        )

    def test_division_by_zero_without_location_names_only_the_metric(self) -> None:
        self.assertEqual(
            str(DivisionByZeroException("Coverage")), "Coverage: division by zero."
        )

    def test_rec_list_too_short_names_the_cutoffs(self) -> None:
        self.assertIn("ats=10", str(RecListTooShortException(10)))

    def test_columns_not_exist_names_the_columns(self) -> None:
        self.assertIn("['A', 'B']", str(ColumnsNotExistException(["A", "B"])))

    def test_segmentation_not_supported_keeps_the_given_message(self) -> None:
        self.assertEqual(str(SegmentationNotSupportedException("too many")), "too many")

    def test_wrong_proportions_has_a_default_message(self) -> None:
        self.assertEqual(
            str(WrongProportionsException()),
            "Proportions doesn't cover all the items/users.",
        )

    def test_invalid_value_names_the_fill_na_value(self) -> None:
        self.assertIn("-1", str(InvalidValueException(-1)))

    def test_invalid_group_names_the_group(self) -> None:
        self.assertEqual(
            str(InvalidGroupException("users")), "users is not a valid group"
        )


if __name__ == "__main__":
    unittest.main()
