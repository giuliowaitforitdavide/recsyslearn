"""Exceptions raised by recsyslearn."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["InvalidInputError", "MissingColumnsError", "RecsyslearnError"]


class RecsyslearnError(Exception):
    """Base class of every error raised by recsyslearn."""


class MissingColumnsError(RecsyslearnError, ValueError):
    """A DataFrame lacks one or more of the columns a function needs."""

    def __init__(self, name: str, missing: Iterable[str]) -> None:
        """Record the ``name`` of the table and the columns it is ``missing``."""
        self.name = name
        self.missing = tuple(missing)
        columns = ", ".join(repr(column) for column in self.missing)
        super().__init__(f"{name} must have the column(s) {columns}")

    def __reduce__(self) -> tuple[type[MissingColumnsError], tuple[str, tuple[str, ...]]]:
        """Rebuild from the two arguments, so that the error can be pickled and copied."""
        return type(self), (self.name, self.missing)


class InvalidInputError(RecsyslearnError, ValueError):
    """The content of an argument violates the definition of what is being computed."""
