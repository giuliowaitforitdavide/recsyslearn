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
opposite order; MovieLens 1M at 20 does (92.2 with descending identifiers). Passing
the popularity as ``score`` does not remove the assumption: it shows that, without an
order, the values are not defined (the input is refused). The
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
from recsyslearn.errors import InvalidInputError

pytestmark = pytest.mark.integration

REPOSITORY = "sisinflab/Top-N-Recommendation-Algorithms-A-Quest-for-the-State-of-the-Art"
COMMIT = "2e6645ed27fbfd7b84f02c50ab9dccc435b60b1a"
FOLDS = range(5)
LENGTH = 20


def read(dataset: str, fold: int, split: str) -> pd.DataFrame:
    path = fetch(REPOSITORY, COMMIT, f"data/{dataset}/splitting/{fold}/{split}.tsv")
    return pd.read_csv(path, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])


def most_popular(
    train: pd.DataFrame, users: np.ndarray, descending_ties: bool, length: int = LENGTH
) -> pd.DataFrame:
    """Top-``length`` most popular unseen items of every user."""
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
        unseen = [item for item in ranking[: length + len(seen[user])] if item not in seen[user]]
        rows += [(user, item, rank) for rank, item in enumerate(unseen[:length], 1)]
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


def scored_most_popular(
    dataset: str, fold: int, length: int = LENGTH
) -> tuple[pd.DataFrame, np.ndarray]:
    """The lists of the baseline with the popularity as ``score``, and the catalog.

    The lists are those of :func:`most_popular` with ascending identifiers among equally
    popular items: that assumed order still decides which items enter a list of
    ``length`` items. The scores only leave the order *within* the list undefined.
    """
    train, test = read(dataset, fold, "train"), read(dataset, fold, "test")
    users = np.intersect1d(train["user"].unique(), test["user"].unique())
    popularity = train.drop_duplicates(["user", "item"])["item"].value_counts()
    ranked = most_popular(train, users, descending_ties=False, length=length)
    scored = ranked.assign(score=ranked["item"].map(popularity)).drop(columns="rank")
    return scored, train["item"].unique()


@pytest.mark.parametrize(
    ("dataset", "printed"),
    [("movielens_1m", 92.0), ("epinions", 72.0)],  # Tables 9 and 11, at 20
)
def test_item_coverage_from_scores(dataset, printed):
    """Lists of 20 items given with scores instead of ranks reproduce the printed value.

    A cutoff equal to the length of the lists splits no tie, so the scores are
    accepted. This does not make the value independent of the order of equally popular
    items: the lists were cut to 20 with the assumed order (see
    :func:`scored_most_popular`), and the next test shows what happens without it."""
    counts = []
    for fold in FOLDS:
        scored, catalog = scored_most_popular(dataset, fold)
        counts.append(coverage(scored, catalog, k=LENGTH) * len(catalog))
    assert round(float(np.mean(counts)), 1) == printed


@pytest.mark.parametrize("dataset", ["movielens_1m", "epinions"])
def test_coverage_at_20_is_refused_when_the_candidates_are_not_cut(dataset):
    """With 40 candidates per user instead of 20, nothing has chosen which of the
    equally popular items are among the first 20: the value at 20 is not defined
    without an order, and recsyslearn refuses to compute it."""
    scored, catalog = scored_most_popular(dataset, fold=0, length=2 * LENGTH)
    with pytest.raises(InvalidInputError, match="same score straddle the cutoff"):
        coverage(scored, catalog, k=LENGTH)


@pytest.mark.parametrize("dataset", ["movielens_1m", "epinions"])
def test_coverage_is_refused_when_a_tie_straddles_the_cutoff(dataset):
    """At 10, some user has equally popular items on both sides of the cutoff: which of
    them are recommended is not defined, and no published definition of coverage says
    what to do, so recsyslearn refuses instead of choosing an order."""
    scored, catalog = scored_most_popular(dataset, fold=0)
    with pytest.raises(InvalidInputError, match="same score straddle the cutoff"):
        coverage(scored, catalog, k=10)
