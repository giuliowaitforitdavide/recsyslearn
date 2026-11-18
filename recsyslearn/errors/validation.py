"""Exceptions raised when an input does not have the expected structure."""

from __future__ import annotations

__all__ = ["ColumnsNotExistException", "InvalidGroupException"]


class ColumnsNotExistException(Exception):
    """Exception raised when columns are not found in the DataFrame."""

    def __init__(
        self, columns: list, message: str = "Dataframe does not contain columns."
    ) -> None:
        super().__init__(f"{message} {columns}")


class InvalidGroupException(ValueError):
    """Exception raised when the users or items to group by are neither 'user' nor 'item'."""

    def __init__(self, user_group: str) -> None:
        super().__init__(f"{user_group} is not a valid group")
