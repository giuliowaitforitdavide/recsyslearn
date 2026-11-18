"""Exceptions raised when a metric cannot be computed for the given input."""

from __future__ import annotations

__all__ = ["DivisionByZeroException", "RecListTooShortException"]


class DivisionByZeroException(ZeroDivisionError):
    """
    Exception raised when a metric divides by zero, i.e. when its value is undefined for the given input.

    It is also a ZeroDivisionError, so code catching ZeroDivisionError keeps working.
    """

    def __init__(self, metric: str, where: str | None = None) -> None:
        location = f" for {where}" if where else ""
        super().__init__(f"{metric}: division by zero{location}.")


class RecListTooShortException(Exception):
    """Exception raised when the values for the recommendation list length is not compatbile with the dataset."""

    def __init__(
        self,
        ats: int | tuple[int, ...],
        message: str = "Values for the recommendation list length is not compatbile with the dataset.",
    ) -> None:
        super().__init__(f"{message} ats={ats}")
