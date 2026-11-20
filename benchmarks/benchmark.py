"""Time every public function on synthetic data.

uv run python benchmarks/benchmark.py [n_users] [n_items] [list_length]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from recsyslearn import accuracy, beyond_accuracy, fairness, segmentation


def synthetic(n_users: int, n_items: int, length: int, seed: int = 0):
    """Training and test interactions with Zipf-like popularity, and top-``length`` lists."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_items + 1)
    weights /= weights.sum()

    def interactions(per_user: int) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "user": np.repeat(np.arange(n_users), per_user),
                "item": rng.choice(n_items, size=n_users * per_user, p=weights),
            }
        )
        return frame.drop_duplicates(ignore_index=True)

    train = interactions(40)
    test = interactions(10)
    test["rank"] = rng.integers(1, 6, size=len(test)).astype(float)
    scores = rng.random((n_users, n_items)) * weights
    top = np.argpartition(-scores, length, axis=1)[:, :length]
    recommendations = pd.DataFrame(
        {
            "user": np.repeat(np.arange(n_users), length),
            "item": top.ravel(),
            "rank": np.tile(np.arange(1, length + 1), n_users),
        }
    )
    return train, test, recommendations


def main() -> None:
    """Time every public function three times and print the best run."""
    sizes = [50_000, 5_000, 20]
    given = [int(arg) for arg in sys.argv[1:4]]
    n_users, n_items, length = given + sizes[len(given) :]
    train, test, recommendations = synthetic(n_users, n_items, length)
    print(
        f"{n_users:,} users, {n_items:,} items: {len(recommendations):,} recommendations, "
        f"{len(train):,} training and {len(test):,} test interactions\n"
    )

    item_groups = segmentation.by_cumulative_interactions(train, (0.2, 0.6, 0.2))
    user_groups = segmentation.by_interaction_count(train, (0.2, 0.8), by="user")
    catalog = np.arange(n_items)
    item_groups = (
        pd.DataFrame({"item": catalog}).merge(item_groups, how="left").fillna({"group": 3})
    )
    user_groups = (
        pd.DataFrame({"user": np.arange(n_users)})
        .merge(user_groups, how="left")
        .fillna({"group": 2})
    )

    cases = {
        "segmentation.by_cumulative_interactions": lambda: segmentation.by_cumulative_interactions(
            train
        ),
        "segmentation.by_interaction_count": lambda: segmentation.by_interaction_count(
            train, by="user"
        ),
        "segmentation.popularity": lambda: segmentation.popularity(train, relative_to="population"),
        "accuracy.ndcg": lambda: accuracy.ndcg(recommendations, test, k=10),
        "beyond_accuracy.coverage": lambda: beyond_accuracy.coverage(recommendations, catalog),
        "beyond_accuracy.novelty": lambda: beyond_accuracy.novelty(recommendations, train),
        "fairness.entropy": lambda: fairness.entropy(recommendations, item_groups),
        "fairness.kl_divergence": lambda: fairness.kl_divergence(
            recommendations, user_groups, ground_truth=test
        ),
        "fairness.mutual_information": lambda: fairness.mutual_information(
            recommendations, user_groups=user_groups, item_groups=item_groups
        ),
    }
    for name, run in cases.items():
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - start)
        print(f"{name:<42} {best * 1000:9.1f} ms")


if __name__ == "__main__":
    main()
