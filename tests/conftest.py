"""Helpers and random data shared by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def random_data():
    """Random recommendations, held-out ratings and training interactions."""
    rng = np.random.default_rng(20261119)
    n_users, n_items, length = 40, 60, 12

    recommendations = []
    for user in range(n_users):
        items = rng.choice(n_items, size=length, replace=False)
        recommendations += [(f"u{user}", f"i{item}", rank) for rank, item in enumerate(items, 1)]
    recommendations = frame(recommendations, ["user", "item", "rank"])

    ratings = set()
    for user in range(n_users - 5):  # five users have no held-out item
        for item in rng.choice(n_items, size=rng.integers(1, 9), replace=False):
            ratings.add((f"u{user}", f"i{item}"))
    ground_truth = frame(sorted(ratings), ["user", "item"])
    ground_truth["rank"] = rng.integers(1, 6, size=len(ground_truth)).astype(float)

    # Zipf-like popularity; pairs can repeat (e.g. play counts).
    weights = 1.0 / np.arange(1, n_items + 1)
    interactions = frame(
        {
            "user": [f"u{user}" for user in rng.integers(0, n_users + 10, size=1500)],
            "item": [
                f"i{item}" for item in rng.choice(n_items, size=1500, p=weights / weights.sum())
            ],
        },
        ["user", "item"],
    )

    user_groups = frame(
        [(f"u{user}", "active" if user % 3 == 0 else "casual") for user in range(n_users)],
        ["user", "group"],
    )
    item_groups = frame(
        [(f"i{item}", ("head", "mid", "tail")[min(item // 10, 2)]) for item in range(n_items)],
        ["item", "group"],
    )
    return {
        "recommendations": recommendations.sample(frac=1, random_state=7).reset_index(drop=True),
        "ground_truth": ground_truth,
        "interactions": interactions,
        "user_groups": user_groups,
        "item_groups": item_groups,
    }
