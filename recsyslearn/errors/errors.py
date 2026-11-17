"""
Deprecated: import the exceptions from the module of their domain instead.

- ``recsyslearn.errors.base``: ``RecsyslearnException``
- ``recsyslearn.errors.validation``: ``ColumnsNotExistException``
- ``recsyslearn.errors.metrics``: ``DivisionByZeroException``, ``RecListTooShortException``
- ``recsyslearn.errors.segmentation``: ``InvalidGroupException``, ``InvalidValueException``,
  ``SegmentationNotSupportedException``, ``WrongProportionsException``

This module will be removed in a future major release.
"""

from __future__ import annotations

import warnings

from recsyslearn.errors.base import RecsyslearnException
from recsyslearn.errors.metrics import DivisionByZeroException, RecListTooShortException
from recsyslearn.errors.segmentation import (
    InvalidGroupException,
    InvalidValueException,
    SegmentationNotSupportedException,
    WrongProportionsException,
)
from recsyslearn.errors.validation import ColumnsNotExistException

__all__ = [
    "ColumnsNotExistException",
    "DivisionByZeroException",
    "InvalidGroupException",
    "InvalidValueException",
    "RecListTooShortException",
    "RecsyslearnException",
    "SegmentationNotSupportedException",
    "WrongProportionsException",
]

warnings.warn(
    "recsyslearn.errors.errors is deprecated: import the exceptions from "
    "recsyslearn.errors.base, recsyslearn.errors.validation, "
    "recsyslearn.errors.metrics or recsyslearn.errors.segmentation instead.",
    DeprecationWarning,
    stacklevel=2,
)
