"""Reproduce the toy example (Tables 1 and 2) of

    Y. Deldjoo, V. W. Anelli, H. Zamani, A. Bellogín, T. Di Noia. "Recommender systems
    fairness evaluation via generalized cross entropy". RMSE workshop @ RecSys 2019.
    https://arxiv.org/abs/1908.06708

Table 1 lists, for six users in two groups, their relevant items and the top-3 lists of
three recommenders; Table 2 prints the generalized cross entropy (GCE) between three
fair distributions and the distribution of the relevant recommendations over the two
user groups. That distribution is what recsyslearn computes
(:func:`recsyslearn.fairness.group_distribution` with flat exposure and a ground truth).
GCE itself is not a metric of recsyslearn: it is evaluated here, from its formula in the
paper, only to compare with the printed numbers.

The data are small enough to be typed in, so this test needs no download.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recsyslearn.fairness import group_distribution

GROUPS = pd.DataFrame({"user": [1, 2, 3, 4, 5, 6], "group": ["a1", "a1", "a1", "a2", "a2", "a2"]})
RELEVANT = {1: [1, 3, 7], 2: [5, 8], 3: [2, 7], 4: [3, 4, 9], 5: [5, 7, 10], 6: [1, 3, 6, 9]}
RECOMMENDED = {
    "Rec 0": [[1, 6, 8], [2, 5, 9], [1, 6, 7], [3, 4, 9], [1, 5, 7], [2, 6, 9]],
    "Rec 1": [[1, 5, 9], [2, 5, 7], [2, 5, 9], [4, 5, 6], [1, 2, 10], [1, 5, 8]],
    "Rec 2": [[1, 3, 7], [1, 5, 8], [2, 7, 9], [3, 4, 9], [5, 7, 10], [3, 6, 9]],
}
# Fair distributions over (a1, a2), and |GCE| printed in Table 2 for each of them.
FAIR = [(1 / 2, 1 / 2), (2 / 3, 1 / 3), (1 / 3, 2 / 3)]
PRINTED = {
    "Rec 0": [0.0800, 0.3025, 0.0025],
    "Rec 1": [0.0, 0.0625, 0.0625],
    "Rec 2": [0.0078, 0.1182, 0.0244],
}
# Share of the relevant recommendations that go to each group, counted by hand in Table 1.
BY_HAND = {"Rec 0": (3 / 10, 7 / 10), "Rec 1": (3 / 6, 3 / 6), "Rec 2": (7 / 16, 9 / 16)}


def generalized_cross_entropy(fair, model, alpha=-1.0):
    """``(sum_j fair_j^alpha * model_j^(1 - alpha) - 1) / (alpha (1 - alpha))``, as in the paper."""
    total = sum(f**alpha * m ** (1 - alpha) for f, m in zip(fair, model, strict=True))
    return (total - 1) / (alpha * (1 - alpha))


def distribution(recommender: str) -> pd.Series:
    truth = pd.DataFrame(
        [(user, item) for user, items in RELEVANT.items() for item in items],
        columns=["user", "item"],
    )
    recommendations = pd.DataFrame(
        [
            (user, item, rank)
            for user, items in enumerate(RECOMMENDED[recommender], 1)
            for rank, item in enumerate(items, 1)
        ],
        columns=["user", "item", "rank"],
    )
    return group_distribution(recommendations, GROUPS, exposure="flat", ground_truth=truth)


@pytest.mark.parametrize("recommender", list(RECOMMENDED))
def test_distribution_of_relevant_recommendations(recommender):
    assert tuple(distribution(recommender)) == pytest.approx(BY_HAND[recommender])


@pytest.mark.parametrize("recommender", list(RECOMMENDED))
def test_generalized_cross_entropy_printed_in_the_paper(recommender):
    model = tuple(distribution(recommender))
    values = [abs(generalized_cross_entropy(fair, model)) for fair in FAIR]
    assert [round(value, 4) for value in values] == PRINTED[recommender]
