"""Reproduce the item coverage of the most-popular baseline in Tables 9 and 11 of

    V. W. Anelli, A. Bellogín, T. Di Noia, D. Jannach, C. Pomo. "Top-N recommendation
    algorithms: a quest for the state-of-the-art". UMAP 2022.
    https://arxiv.org/abs/2203.01155

from the five train/test folds the authors publish for MovieLens 1M and Epinions
(https://github.com/sisinflab/Top-N-Recommendation-Algorithms-A-Quest-for-the-State-of-the-Art).

The paper reports "IC", the *number* of distinct items recommended in the top 10 and top
20, averaged over the folds: it is the catalog coverage of recsyslearn multiplied by the
size of the catalog (the items of the training fold).

The recommender is deterministic: every user is recommended the items with the most
training users among those the user has not interacted with. The only choice the paper
does not document is the order of equally popular items; ascending identifier is used
here. Three of the four values do not depend on that choice, which is checked with the
opposite order; MovieLens 1M at 20 does (92.2 with descending identifiers). The
Amazon Music table of the same paper is left out because its identifiers are strings,
whose order in the framework is not reproducible.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from paper_data import fetch

from recsyslearn.beyond_accuracy import coverage

pytestmark = pytest.mark.integration

REPOSITORY = "sisinflab/Top-N-Recommendation-Algorithms-A-Quest-for-the-State-of-the-Art"
COMMIT = "2e6645ed27fbfd7b84f02c50ab9dccc435b60b1a"
FOLDS = range(5)
LENGTH = 20


def read(dataset: str, fold: int, split: str) -> pd.DataFrame:
    path = fetch(REPOSITORY, COMMIT, f"data/{dataset}/splitting/{fold}/{split}.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=["user", "item", "rank", "timestamp"])


def most_popular(train: pd.DataFrame, users: np.ndarray, descending_ties: bool) -> pd.DataFrame:
    """Top-``LENGTH`` most popular unseen items of every user."""
    pairs = train.drop_duplicates(["user", "item"])
    popularity = pairs["item"].value_counts()
    ranking = (
        popularity.rename("popularity")
        .rename_axis("item")
        .reset_index()
        .sort_values(["popularity", "item"], ascending=[False, not descending_ties])["item"]
        .to_numpy()
    )
    seen = pairs.groupby("user")["item"].agg(set)
    rows = []
    for user in users:
        unseen = [item for item in ranking[: LENGTH + len(seen[user])] if item not in seen[user]]
        rows += [(user, item, rank) for rank, item in enumerate(unseen[:LENGTH], 1)]
    return pd.DataFrame(rows, columns=["user", "item", "rank"])


@functools.cache
def recommended_items(dataset: str, k: int, descending_ties: bool) -> float:
    """Number of distinct recommended items in the top ``k``, averaged over the folds."""
    counts = []
    for fold in FOLDS:
        train, test = read(dataset, fold, "train"), read(dataset, fold, "test")
        # The framework evaluates the users who have at least one test item.
        users = np.intersect1d(train["user"].unique(), test["user"].unique())
        recommendations = most_popular(train, users, descending_ties)
        catalog = train["item"].unique()
        counts.append(coverage(recommendations, catalog, k=k) * len(catalog))
    return float(np.mean(counts))


@pytest.mark.parametrize(
    ("dataset", "k", "printed"),
    [
        ("movielens_1m", 10, 56.2),  # Table 9
        ("movielens_1m", 20, 92.0),
        ("epinions", 10, 41.6),  # Table 11
        ("epinions", 20, 72.0),
    ],
)
def test_item_coverage_of_most_popular(dataset, k, printed):
    assert round(recommended_items(dataset, k, descending_ties=False), 1) == printed


@pytest.mark.parametrize(
    ("dataset", "k", "printed"),
    [("movielens_1m", 10, 56.2), ("epinions", 10, 41.6), ("epinions", 20, 72.0)],
)
def test_result_does_not_depend_on_the_order_of_ties(dataset, k, printed):
    assert round(recommended_items(dataset, k, descending_ties=True), 1) == printed


def test_movielens_at_20_depends_on_the_order_of_ties():
    """How far the fourth value moves with the opposite order: 92.2 instead of 92.0."""
    assert round(recommended_items("movielens_1m", 20, descending_ties=True), 1) == 92.2
