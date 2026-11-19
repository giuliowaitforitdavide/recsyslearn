"""Exceptions raised when users or items cannot be segmented into groups."""

from __future__ import annotations

__all__ = [
    "InvalidValueException",
    "SegmentationNotSupportedException",
    "WrongProportionsException",
]


class SegmentationNotSupportedException(Exception):
    """Exception raised when there are too many group to be segmented."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WrongProportionsException(Exception):
    """Exception raised when proportions of the segmentation does not cover all the items or users."""

    def __init__(
        self, message: str = "Proportions doesn't cover all the items/users."
    ) -> None:
        super().__init__(message)


class InvalidValueException(Exception):
    """Exception raised when the value of fill_na of the segmentation value is invalid."""

    def __init__(self, fill_na: object) -> None:
        message = (
            f"Feature contains {fill_na} as value. Please select another fill_na value."
        )
        super().__init__(message)
