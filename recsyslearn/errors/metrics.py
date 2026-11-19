"""Exceptions raised when a metric cannot be computed for the given input."""

from __future__ import annotations

__all__ = [
    "DivisionByZeroException",
    "InvalidCutoffException",
    "RecListTooShortException",
]


class DivisionByZeroException(ZeroDivisionError):
    """
    Exception raised when a metric divides by zero, i.e. when its value is undefined for the given input.

    It is also a ZeroDivisionError, so code catching ZeroDivisionError keeps working.
    """

    def __init__(self, metric: str, where: str | None = None) -> None:
        location = f" for {where}" if where else ""
        super().__init__(f"{metric}: division by zero{location}.")


class RecListTooShortException(ValueError):
    """Exception raised when a cut-off is longer than the shortest recommendation list."""

    def __init__(self, metric: str, k: int, shortest: int) -> None:
        super().__init__(
            f"{metric}@{k} needs recommendation lists of at least {k} items: "
            f"the shortest has {shortest}."
        )


class InvalidCutoffException(ValueError):
    """Exception raised when the cut-off of a metric is not a positive integer."""

    def __init__(self, metric: str, k: object) -> None:
        super().__init__(
            f"{metric}: the cut-off k must be a positive integer, got {k!r}."
        )
