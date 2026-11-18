"""
Hypothesis strategies that generate valid inputs for the metrics.

They are the building blocks of the property-based tests (see AGENTS.md, section 4): every generated
DataFrame follows the data contract of AGENTS.md, section 3, and is small enough to keep tests fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

MAX_USERS = 8
MAX_ITEMS = 20
MAX_GROUPS = 4


def _ids(prefix: str, count: int) -> list[str]:
    """
    Build string ids such as "u1", "u2", … .

    :param prefix: Prefix of every id.
    :type prefix: str
    :param count: Number of ids.
    :type count: int
    :return: The ids, from 1 to count.
    :rtype: list[str]
    """

    return [f"{prefix}{index}" for index in range(1, count + 1)]


@st.composite
def catalogues(draw: st.DrawFn, min_items: int = 1) -> list[str]:
    """
    Generate an item catalogue.

    :param draw: Hypothesis draw function.
    :type draw: st.DrawFn
    :param min_items: Minimum number of items.
    :type min_items: int, default 1
    :return: Distinct item ids.
    :rtype: list[str]
    """

    return _ids("i", draw(st.integers(min_value=min_items, max_value=MAX_ITEMS)))


@st.composite
def top_n(
    draw: st.DrawFn,
    items: list[str] | None = None,
    min_length: int = 1,
    max_length: int = 10,
) -> pd.DataFrame:
    """
    Generate recommendation lists in the form (user, item, rank).

    Every user gets distinct items ranked from 1 to the length of their list; lists may have
    different lengths.

    :param draw: Hypothesis draw function.
    :type draw: st.DrawFn
    :param items: The item catalogue to draw from; generated if None.
    :type items: list[str] or None, default None
    :param min_length: Minimum length of a list.
    :type min_length: int, default 1
    :param max_length: Maximum length of a list (capped by the catalogue size).
    :type max_length: int, default 10
    :return: A DataFrame with columns user, item, rank.
    :rtype: pd.DataFrame
    """

    items = items if items is not None else draw(catalogues(min_items=min_length))
    users = _ids("u", draw(st.integers(min_value=1, max_value=MAX_USERS)))
    longest = min(max_length, len(items))
    rows = []
    for user in users:
        length = draw(
            st.integers(min_value=min(min_length, longest), max_value=longest)
        )
        ranked = draw(st.permutations(items))[:length]
        rows.extend([user, item, rank] for rank, item in enumerate(ranked, start=1))
    return pd.DataFrame(rows, columns=["user", "item", "rank"])


@st.composite
def pos_items(
    draw: st.DrawFn, users: list[str], items: list[str], allow_empty: bool = False
) -> pd.DataFrame:
    """
    Generate the relevant items of each user in the form (user, pos_items).

    :param draw: Hypothesis draw function.
    :type draw: st.DrawFn
    :param users: The users that get relevant items.
    :type users: list[str]
    :param items: The item catalogue to draw from.
    :type items: list[str]
    :param allow_empty: Whether a user may have no relevant items.
    :type allow_empty: bool, default False
    :return: A DataFrame with columns user, pos_items (an array of item ids).
    :rtype: pd.DataFrame
    """

    rows = []
    for user in users:
        relevant = draw(
            st.lists(
                st.sampled_from(items),
                min_size=0 if allow_empty else 1,
                max_size=len(items),
                unique=True,
            )
        )
        rows.append([user, np.array(relevant, dtype=object)])
    return pd.DataFrame(rows, columns=["user", "pos_items"])


@st.composite
def groups(draw: st.DrawFn, ids: list[str], key: str) -> pd.DataFrame:
    """
    Assign every user or item to a group, in the form (key, group).

    :param draw: Hypothesis draw function.
    :type draw: st.DrawFn
    :param ids: The users or items to assign.
    :type ids: list[str]
    :param key: Name of the id column, "user" or "item".
    :type key: str
    :return: A DataFrame with columns key, group.
    :rtype: pd.DataFrame
    """

    names = _ids("g", draw(st.integers(min_value=1, max_value=MAX_GROUPS)))
    assigned = [draw(st.sampled_from(names)) for _ in ids]
    return pd.DataFrame({key: ids, "group": assigned})


@st.composite
def target_representations(draw: st.DrawFn, group_names: list[str]) -> pd.DataFrame:
    """
    Generate a strictly positive target representation that sums to 1 over the given groups.

    :param draw: Hypothesis draw function.
    :type draw: st.DrawFn
    :param group_names: The groups that get a target share.
    :type group_names: list[str]
    :return: A DataFrame with columns group, target_representation.
    :rtype: pd.DataFrame
    """

    weights = [
        draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False))
        for _ in group_names
    ]
    total = sum(weights)
    return pd.DataFrame(
        {
            "group": group_names,
            "target_representation": [weight / total for weight in weights],
        }
    )
