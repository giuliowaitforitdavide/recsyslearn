"""Evaluation of recommender systems: accuracy, beyond-accuracy and fairness metrics,
and segmentations of users and items.

Every function works on pandas DataFrames with the columns ``["user", "item"]``
(implicit feedback) or ``["user", "item", "rank"]`` (explicit feedback), and is
implemented from its definition in the literature by a native Rust core.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from recsyslearn import accuracy, beyond_accuracy, errors, fairness, segmentation

__all__ = ["accuracy", "beyond_accuracy", "errors", "fairness", "segmentation"]

__author__ = "Giulio Davide Carparelli"
__email__ = "giuliowaitforitdavide@duck.com"

try:
    __version__ = version("recsyslearn")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
