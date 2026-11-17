"""Exceptions raised when users or items cannot be segmented into groups."""

from __future__ import annotations

from recsyslearn.errors.base import RecsyslearnException

__all__ = [
    "InvalidGroupException",
    "InvalidValueException",
    "SegmentationNotSupportedException",
    "WrongProportionsException",
]


class SegmentationNotSupportedException(RecsyslearnException):
    """Exception raised when there are too many group to be segmented."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WrongProportionsException(RecsyslearnException):
    """Exception raised when proportions of the segmentation does not cover all the items or users."""

    def __init__(
        self, message: str = "Proportions doesn't cover all the items/users."
    ) -> None:
        super().__init__(message)


class InvalidValueException(RecsyslearnException):
    """Exception raised when the value of fill_na of the segmentation value is invalid."""

    def __init__(self, fill_na) -> None:
        message = (
            f"Feature contains {fill_na} as value. Please select another fill_na value."
        )
        super().__init__(message)


class InvalidGroupException(RecsyslearnException):
    """Exception raised when user wants to segment a non valid group"""

    def __init__(self, user_group) -> None:
        super().__init__(f"{user_group} is not a valid group")
