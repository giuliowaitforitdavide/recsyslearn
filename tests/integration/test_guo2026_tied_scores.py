"""Reproduce the toy example of Section 3.2 of

    C. Guo, H. Chen, Y. Zhu, Y. Li. "Tie handling is part of the evaluation protocol: an
    order-invariance audit for tie-heavy recommender scores". FRAME workshop @ RecSys
    2026. https://arxiv.org/abs/2609.26977

One relevant item and 30 negatives all receive the same score, and the list is cut at
10. The paper prints the value that "uniform random tie-breaking" gives in expectation,
NDCG@10 = 0.1466, against 1 when the relevant item is placed first. It is the value of
the tie-aware nDCG of McSherry & Najork (2008) that recsyslearn computes for a ``score``
column with ties; the paper's Eq. 2 gives it in closed form for one relevant item with
``a`` candidates above it and a tie block of size ``t``:

    E[NDCG@k] = (1 / t) * sum_{r = a + 1}^{min(a + t, k)} 1 / log2(r + 1)

The data are small enough to be typed in, so this test needs no download.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from recsyslearn.accuracy import ndcg

TRUTH = pd.DataFrame({"user": ["u"], "item": [0]})


def candidates(above: int, tied: int) -> pd.DataFrame:
    """Item 0, relevant, in a block of ``tied`` equal scores below ``above`` better ones."""
    scores = [2.0 + position for position in range(above)] + [1.0] * tied
    items = list(range(tied, tied + above)) + list(range(tied))
    return pd.DataFrame({"user": "u", "item": items, "score": scores})


def test_printed_value():
    """Section 3.2: a = 0, t = 31, k = 10 gives an expected NDCG@10 of 0.1466."""
    assert round(ndcg(candidates(above=0, tied=31), TRUTH, k=10), 4) == 0.1466


def test_placing_the_relevant_item_first_gives_one():
    """The other value of the example: the optimistic order, given as explicit ranks."""
    ranked = pd.DataFrame({"user": "u", "item": range(31), "rank": range(1, 32)})
    assert ndcg(ranked, TRUTH, k=10) == 1.0


@pytest.mark.parametrize(
    ("above", "tied", "k"),
    [(0, 31, 10), (3, 5, 10), (8, 5, 10), (10, 4, 10), (0, 1, 1), (2, 7, 3)],
)
def test_closed_form_of_equation_2(above, tied, k):
    expected = sum(1 / math.log2(r + 1) for r in range(above + 1, min(above + tied, k) + 1)) / tied
    assert ndcg(candidates(above, tied), TRUTH, k=k) == pytest.approx(expected, abs=1e-12)
