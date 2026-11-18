"""Exceptions raised when an input does not have the expected structure."""

from __future__ import annotations

__all__ = ["ColumnsNotExistException"]


class ColumnsNotExistException(Exception):
    """Exception raised when columns are not found in the DataFrame."""

    def __init__(
        self, columns: list, message: str = "Dataframe does not contain columns."
    ) -> None:
        super().__init__(f"{message} {columns}")
