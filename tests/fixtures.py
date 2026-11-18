"""
Builders of the test fixtures.

Every builder returns a new DataFrame on each call, so a test that mutates its input cannot affect
other tests (see AGENTS.md, section 4). Data stays small and explicit, so that expected values can
be computed by hand.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from recsyslearn.dataset.utils import find_relevant_items


def make_top_n(
    lists: dict[str, Sequence[str]], rank: float | None = None
) -> pd.DataFrame:
    """
    Build recommendation lists or interactions in the form (user, item, rank).

    :param lists: The items of each user, in order.
    :type lists: dict[str, Sequence[str]]
    :param rank: The rank of every row. If None, the rank is the position in the user's list (1 = top).
    :type rank: float or None, default None
    :return: A new DataFrame with columns user, item, rank, ordered by user and then by position.
    :rtype: pd.DataFrame
    """

    rows = [
        [user, item, position if rank is None else rank]
        for user, items in lists.items()
        for position, item in enumerate(items, start=1)
    ]
    return pd.DataFrame(rows, columns=["user", "item", "rank"])


def make_scores(scores: dict[str, Sequence[tuple[str, float]]]) -> pd.DataFrame:
    """
    Build rows (user, item, rank) where the rank of each item is given explicitly.

    :param scores: The (item, rank) pairs of each user, in order.
    :type scores: dict[str, Sequence[tuple[str, float]]]
    :return: A new DataFrame with columns user, item, rank.
    :rtype: pd.DataFrame
    """

    rows = [
        [user, item, rank] for user, pairs in scores.items() for item, rank in pairs
    ]
    return pd.DataFrame(rows, columns=["user", "item", "rank"])


def make_mapping(values: dict[str, object], key: str, value: str) -> pd.DataFrame:
    """
    Build a two-column DataFrame mapping each key to a value (e.g. item → group).

    :param values: The value of each key, in order.
    :type values: dict[str, object]
    :param key: Name of the key column.
    :type key: str
    :param value: Name of the value column.
    :type value: str
    :return: A new DataFrame with columns key, value.
    :rtype: pd.DataFrame
    """

    return pd.DataFrame(list(values.items()), columns=[key, value])


def item_groups() -> pd.DataFrame:
    """Ten items in three groups: 2 items in group 1, 3 in group 2, 5 in group 3."""
    return make_mapping(
        {
            "1": "1",
            "2": "1",
            "3": "2",
            "4": "2",
            "5": "2",
            "6": "3",
            "7": "3",
            "8": "3",
            "9": "3",
            "10": "3",
        },
        key="item",
        value="group",
    )


def user_groups() -> pd.DataFrame:
    """Six users in two groups: users 1 and 2 in group 1, users 3 to 6 in group 2."""
    return make_mapping(
        {"1": "1", "2": "1", "3": "2", "4": "2", "5": "2", "6": "2"},
        key="user",
        value="group",
    )


def top_5() -> pd.DataFrame:
    """Top-5 recommendation lists for six users, ranked by position."""
    return make_top_n(
        {
            "1": ["2", "8", "5", "3", "6"],
            "2": ["1", "2", "6", "7", "9"],
            "3": ["6", "5", "8", "7", "9"],
            "4": ["1", "2", "3", "4", "5"],
            "5": ["5", "4", "3", "2", "1"],
            "6": ["9", "8", "6", "3", "4"],
        }
    )


def top_5_unranked() -> pd.DataFrame:
    """Five items per user for six users, all with rank 1, so every item has exposure 1."""
    return make_top_n(
        {
            "1": ["3", "4", "6", "7", "8"],
            "2": ["1", "2", "5", "6", "7"],
            "3": ["2", "3", "6", "9", "10"],
            "4": ["1", "3", "6", "7", "9"],
            "5": ["1", "3", "5", "7", "9"],
            "6": ["2", "3", "5", "9", "10"],
        },
        rank=1,
    )


def top_6() -> pd.DataFrame:
    """Top-6 recommendation lists for six users, with explicit ranks from 1 to 6 not in row order."""
    return make_scores(
        {
            "1": [("1", 1), ("3", 6), ("4", 3), ("5", 4), ("8", 5), ("9", 2)],
            "2": [("2", 6), ("3", 5), ("6", 4), ("7", 3), ("8", 2), ("9", 1)],
            "3": [("2", 1), ("3", 2), ("4", 3), ("7", 4), ("8", 5), ("9", 6)],
            "4": [("1", 4), ("3", 5), ("4", 6), ("7", 3), ("8", 2), ("9", 1)],
            "5": [("1", 6), ("3", 2), ("5", 3), ("6", 4), ("8", 5), ("9", 1)],
            "6": [("2", 3), ("3", 5), ("5", 4), ("6", 6), ("8", 2), ("9", 1)],
        }
    )


def binary_relevance() -> pd.DataFrame:
    """Relevant items with binary relevance (1) of users 1, 2, 3, 4 and 6; user 5 has none."""
    return make_top_n(
        {
            "1": ["2", "9"],
            "2": ["6", "7"],
            "3": ["1", "7", "9"],
            "4": ["3", "7"],
            "6": ["2", "9"],
        },
        rank=1,
    )


def graded_relevance() -> pd.DataFrame:
    """Relevant items of all six users with graded relevance scores (0.143 or 0.077)."""
    return make_scores(
        {
            "1": [("2", 0.143), ("9", 0.077)],
            "2": [("6", 0.077), ("7", 0.077)],
            "3": [
                ("1", 0.143),
                ("3", 0.143),
                ("4", 0.143),
                ("5", 0.143),
                ("7", 0.077),
                ("9", 0.077),
            ],
            "4": [
                ("2", 0.143),
                ("3", 0.143),
                ("5", 0.07743),
                ("7", 0.077),
                ("8", 0.077),
                ("9", 0.077),
            ],
            "5": [
                ("1", 0.143),
                ("2", 0.143),
                ("4", 0.143),
                ("5", 0.143),
                ("6", 0.077),
                ("9", 0.077),
            ],
            "6": [
                ("1", 0.143),
                ("2", 0.143),
                ("6", 0.077),
                ("8", 0.077),
                ("9", 0.077),
            ],
        }
    )


def pos_items() -> pd.DataFrame:
    """Relevant items of all six users in the form (user, pos_items); user 6 has a single relevant item."""
    relevance = make_top_n(
        {
            "1": ["2", "9"],
            "2": ["6", "7"],
            "3": ["1", "3", "4", "5", "7", "9"],
            "4": ["2", "3", "5", "7", "8", "9"],
            "5": ["1", "2", "4", "5", "6", "9"],
            "6": ["9"],
        },
        rank=1,
    )
    return find_relevant_items(relevance)


def item_interactions() -> pd.DataFrame:
    """Five interactions: item 1 by users 1, 2 and 3, item 2 twice by user 1."""
    return pd.DataFrame(
        [["1", "1", 1], ["2", "1", 1], ["3", "1", 1], ["1", "2", 1], ["1", "2", 1]],
        columns=["user", "item", "rank"],
    )


def user_interactions() -> pd.DataFrame:
    """User 1 interacts four times with items 1 to 9, user 2 twice and user 3 once with item 1."""
    items = [str(item) for item in range(1, 10)]
    return make_top_n({"1": items * 4, "2": ["1", "1"], "3": ["1"]}, rank=1)


def popularity_interactions() -> pd.DataFrame:
    """Ten interactions: item 1 six times, item 2 three times, item 3 once."""
    return pd.DataFrame(
        [
            ["1", "1", 1],
            ["1", "1", 1],
            ["1", "1", 1],
            ["1", "1", 1],
            ["2", "1", 1],
            ["3", "1", 1],
            ["1", "2", 1],
            ["1", "2", 1],
            ["2", "2", 1],
            ["1", "3", 1],
        ],
        columns=["user", "item", "rank"],
    )


def item_popularity() -> pd.DataFrame:
    """Popularity percentage of ten items, in the form (item, percentage)."""
    return make_mapping(
        {
            "1": 0.05,
            "2": 0.01,
            "3": 0.03,
            "4": 0.23,
            "5": 0.4,
            "6": 0.6,
            "7": 0.15,
            "8": 0.34,
            "9": 0.07,
            "10": 0.02,
        },
        key="item",
        value="percentage",
    )


def item_genres() -> pd.DataFrame:
    """Genre of four items."""
    return make_mapping(
        {"1": "pop", "2": "pop", "3": "electronic", "4": "rock"},
        key="item",
        value="genre",
    )


def user_genders() -> pd.DataFrame:
    """Gender of four users, two of them missing (None and NaN)."""
    return make_mapping(
        {"1": "m", "2": "f", "3": None, "4": np.nan}, key="user", value="gender"
    )


def user_ages() -> pd.DataFrame:
    """Age of three users; user 3 has -1, the default fill_na value of DiscreteFeatureSegmentation."""
    return make_mapping({"1": 19, "2": 30, "3": -1}, key="user", value="age")
