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


class InvalidInputError(RecsyslearnError, ValueError):
    """The content of an argument violates the definition of what is being computed."""
