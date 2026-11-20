"""Reproduce the nDCG of the most-popular baseline in Tables 8 and 9 of

    M. Ferrari Dacrema, S. Boglio, P. Cremonesi, D. Jannach. "A troubling analysis of
    reproducibility and progress in recommender systems research". ACM Transactions on
    Information Systems 39(2), 2021. https://arxiv.org/abs/1911.07698

from the evaluation data published with

    X. He, L. Liao, H. Zhang, L. Nie, X. Hu, T.-S. Chua. "Neural collaborative
    filtering". WWW 2017. https://github.com/hexiangnan/neural_collaborative_filtering

Protocol (leave-one-out): each user has one held-out item, which is ranked among 99
sampled items the user did not interact with; "TopPopular" orders these candidates by
their number of training interactions. With a single relevant item,
nDCG@k = 1 / log2(1 + position) if the held-out item is in the top k, else 0.

The papers do not say how candidates with the same popularity are ordered, and the
fourth decimal of the result depends on it. The test therefore brackets the printed
value between the two extreme orders (held-out item last, or first, among its ties): the
bracket is less than 0.001 wide, so the first three decimals are reproduced regardless.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from paper_data import fetch

from recsyslearn.accuracy import ndcg

pytestmark = pytest.mark.integration

REPOSITORY = "hexiangnan/neural_collaborative_filtering"
COMMIT = "4aab159e81c44b062c091bdaed0ab54ac632371f"


@functools.cache
def candidates(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out pairs, and the candidates of every user with their popularity."""
    train = pd.read_csv(
        fetch(REPOSITORY, COMMIT, f"Data/{dataset}.train.rating"),
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["user", "item"],
    ).drop_duplicates()  # implicit feedback: a repeated pair is one interaction
    held_out = pd.read_csv(
        fetch(REPOSITORY, COMMIT, f"Data/{dataset}.test.rating"),
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["user", "item"],
    )

    # One line per user: "(user,held-out item)" followed by the 99 sampled items.
    negatives = pd.read_csv(
        fetch(REPOSITORY, COMMIT, f"Data/{dataset}.test.negative"), sep="\t", header=None
    )
    assert negatives.shape == (len(held_out), 100)
    sampled = pd.DataFrame(
        {
            "user": np.repeat(held_out["user"].to_numpy(), 99),
            "item": negatives.iloc[:, 1:].to_numpy().ravel(),
        }
    )
    pool = pd.concat([sampled, held_out], ignore_index=True).drop_duplicates()
    popularity = train["item"].value_counts()
    pool["popularity"] = pool["item"].map(popularity).fillna(0).astype(int)
    pool = pool.merge(held_out.assign(held_out=True), how="left", on=["user", "item"])
    pool["held_out"] = pool["held_out"].notna()
    return held_out, pool


@functools.cache
def top_popular_ndcg(dataset: str, k: int, held_out_first: bool) -> float:
    held_out, pool = candidates(dataset)
    ranked = pool.sort_values(
        ["user", "popularity", "held_out"], ascending=[True, False, not held_out_first]
    )
    # Without a "rank" column, the order of the rows is the ranking.
    return ndcg(ranked[["user", "item"]], held_out, k=k)


@pytest.mark.parametrize(
    ("dataset", "k", "printed"),
    [
        ("ml-1m", 1, 0.1051),  # Table 8
        ("ml-1m", 5, 0.2064),
        ("ml-1m", 10, 0.2542),
        ("pinterest-20", 1, 0.0467),  # Table 9
        ("pinterest-20", 5, 0.1064),
        ("pinterest-20", 10, 0.1409),
    ],
)
def test_ndcg_of_top_popular(dataset, k, printed):
    pessimistic = top_popular_ndcg(dataset, k, held_out_first=False)
    optimistic = top_popular_ndcg(dataset, k, held_out_first=True)
    assert optimistic - pessimistic < 1e-3
    # The printed value has four decimals: allow half a unit of the last one.
    assert pessimistic - 5e-5 <= printed <= optimistic + 5e-5
