"""nDCG@k against the formulas of Järvelin & Kekäläinen (2002), Burges et al. (2005)
and Krichene & Rendle (2020, Eq. 8)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from conftest import frame
from reference import lists_from_rows
from reference import ndcg as reference_ndcg

from recsyslearn.accuracy import ndcg
from recsyslearn.errors import InvalidInputError, MissingColumnsError


def recommend(user, items):
    return [(user, item, rank) for rank, item in enumerate(items, 1)]


RECOMMENDATIONS = frame(recommend("u", "abcde"), ["user", "item", "rank"])


class TestBinaryRelevance:
    """rec = [a, b, c, d, e], Rel = {b, e, z}: hits at positions 2 and 5."""

    truth = frame([("u", "b"), ("u", "e"), ("u", "z")], ["user", "item"])
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)

    def test_ideal_uses_min_k_relevant_positions(self):
        assert ndcg(RECOMMENDATIONS, self.truth, k=5) == pytest.approx(self.dcg / self.idcg)
        assert ndcg(RECOMMENDATIONS, self.truth, k=5) == pytest.approx(0.477624, abs=1e-6)

    def test_ideal_is_truncated_at_k(self):
        expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        assert ndcg(RECOMMENDATIONS, self.truth, k=2) == pytest.approx(expected)
        assert ndcg(RECOMMENDATIONS, self.truth, k=2) == pytest.approx(0.386853, abs=1e-6)

    def test_k_beyond_the_list_adds_nothing(self):
        assert ndcg(RECOMMENDATIONS, self.truth, k=10) == pytest.approx(0.477624, abs=1e-6)
        assert ndcg(RECOMMENDATIONS, self.truth) == pytest.approx(0.477624, abs=1e-6)

    def test_gain_is_irrelevant(self):
        linear = ndcg(RECOMMENDATIONS, self.truth, k=5, gain="linear")
        exponential = ndcg(RECOMMENDATIONS, self.truth, k=5, gain="exponential")
        assert linear == pytest.approx(exponential)

    def test_short_list_is_not_padded(self):
        recommendations = frame(recommend("u", "qb"), ["user", "item", "rank"])
        truth = frame([("u", "b"), ("u", "e")], ["user", "item"])
        assert ndcg(recommendations, truth, k=5) == pytest.approx(0.386853, abs=1e-6)


class TestGradedRelevance:
    """Ground truth {a: 3, b: 2, c: 1, x: 3, y: 2}: the ideal gains are [3, 3, 2, 2, 1]."""

    truth = frame(
        [("u", "a", 3), ("u", "b", 2), ("u", "c", 1), ("u", "x", 3), ("u", "y", 2)],
        ["user", "item", "rank"],
    )

    @pytest.mark.parametrize(
        ("k", "gain", "rounded"),
        [
            (3, "linear", 0.808082),
            (3, "exponential", 0.727193),
            (5, "linear", 0.666834),
            (5, "exponential", 0.643545),
        ],
    )
    def test_matches_the_formula(self, k, gain, rounded):
        def discounted(relevances):
            gains = [r if gain == "linear" else 2**r - 1 for r in relevances[:k]]
            return sum(g / math.log2(position + 1) for position, g in enumerate(gains, 1))

        # The list [a, b, c, d, e] has relevances [3, 2, 1, 0, 0]; the ideal one [3, 3, 2, 2, 1].
        expected = discounted([3, 2, 1, 0, 0]) / discounted([3, 3, 2, 2, 1])
        score = ndcg(RECOMMENDATIONS, self.truth, k=k, gain=gain)
        assert score == pytest.approx(expected)
        assert score == pytest.approx(rounded, abs=1e-6)

    def test_linear_gain_by_hand(self):
        dcg = 3 + 2 / math.log2(3) + 1 / math.log2(4)
        idcg = 3 + 3 / math.log2(3) + 2 / math.log2(4)
        assert ndcg(RECOMMENDATIONS, self.truth, k=3) == pytest.approx(dcg / idcg)

    def test_exponential_gain_by_hand(self):
        dcg = 7 + 3 / math.log2(3) + 1 / math.log2(4)
        idcg = 7 + 7 / math.log2(3) + 3 / math.log2(4)
        assert ndcg(RECOMMENDATIONS, self.truth, k=3, gain="exponential") == pytest.approx(
            dcg / idcg
        )

    def test_perfect_ranking_scores_one(self):
        recommendations = frame(recommend("u", "axbyc"), ["user", "item", "rank"])
        assert ndcg(recommendations, self.truth, gain="exponential") == pytest.approx(1.0)


class TestInputHandling:
    truth = frame([("u", "b"), ("u", "e"), ("u", "z")], ["user", "item"])

    def test_rows_are_ordered_by_rank(self):
        shuffled = RECOMMENDATIONS.sample(frac=1, random_state=3)
        assert ndcg(shuffled, self.truth, k=5) == pytest.approx(0.477624, abs=1e-6)

    def test_row_order_is_the_ranking_without_a_rank_column(self):
        implicit = RECOMMENDATIONS[["user", "item"]]
        assert ndcg(implicit, self.truth, k=5) == pytest.approx(0.477624, abs=1e-6)
        assert ndcg(implicit.iloc[::-1], self.truth, k=5) != pytest.approx(0.477624, abs=1e-6)

    def test_identifiers_can_be_integers(self):
        recommendations = frame([(1, 10, 1), (1, 11, 2)], ["user", "item", "rank"])
        truth = frame([(1, 11)], ["user", "item"])
        assert ndcg(recommendations, truth) == pytest.approx(1 / math.log2(3))

    def test_extra_columns_are_ignored(self):
        recommendations = RECOMMENDATIONS.assign(score=0.5)
        assert ndcg(recommendations, self.truth.assign(timestamp=1), k=5) == pytest.approx(
            0.477624, abs=1e-6
        )

    def test_inputs_are_not_modified(self):
        recommendations, truth = RECOMMENDATIONS.copy(), self.truth.copy()
        ndcg(recommendations, truth, k=5)
        pd.testing.assert_frame_equal(recommendations, RECOMMENDATIONS)
        pd.testing.assert_frame_equal(truth, self.truth)


class TestAveraging:
    recommendations = frame(
        recommend("hit", "ab") + recommend("miss", "ab") + recommend("unjudged", "ab"),
        ["user", "item", "rank"],
    )
    truth = frame([("hit", "b"), ("miss", "c"), ("no-list", "a")], ["user", "item"])

    def test_per_user_scores(self):
        scores = ndcg(self.recommendations, self.truth, k=2, per_user=True)
        assert isinstance(scores, pd.Series)
        assert scores.index.name == "user"
        assert scores.name == "ndcg"
        assert scores.to_dict() == pytest.approx(
            {"hit": 1 / math.log2(3), "miss": 0.0, "no-list": 0.0}
        )

    def test_users_without_relevant_items_are_excluded(self):
        scores = ndcg(self.recommendations, self.truth, k=2, per_user=True)
        assert "unjudged" not in scores.index

    def test_mean_is_over_the_evaluated_users(self):
        expected = (1 / math.log2(3)) / 3
        assert ndcg(self.recommendations, self.truth, k=2) == pytest.approx(expected)


class TestErrors:
    truth = frame([("u", "b")], ["user", "item"])

    def test_missing_columns(self):
        with pytest.raises(MissingColumnsError, match="item"):
            ndcg(RECOMMENDATIONS[["user", "rank"]], self.truth)
        with pytest.raises(MissingColumnsError, match="user"):
            ndcg(RECOMMENDATIONS, self.truth[["item"]])

    def test_not_a_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            ndcg([("u", "a", 1)], self.truth)

    @pytest.mark.parametrize("k", [0, -1, 1.5, "10", True])
    def test_invalid_k(self, k):
        with pytest.raises(InvalidInputError, match="k must be a positive integer"):
            ndcg(RECOMMENDATIONS, self.truth, k=k)

    def test_invalid_gain(self):
        with pytest.raises(InvalidInputError, match="gain"):
            ndcg(RECOMMENDATIONS, self.truth, gain="quadratic")

    def test_duplicate_recommendation(self):
        duplicated = frame([("u", "a", 1), ("u", "a", 2)], ["user", "item", "rank"])
        with pytest.raises(InvalidInputError, match="more than once"):
            ndcg(duplicated, self.truth)

    def test_duplicate_ground_truth(self):
        with pytest.raises(InvalidInputError, match="more than once"):
            ndcg(RECOMMENDATIONS, pd.concat([self.truth, self.truth]))

    def test_negative_relevance(self):
        with pytest.raises(InvalidInputError, match="non-negative"):
            ndcg(RECOMMENDATIONS, self.truth.assign(rank=-1.0))

    def test_missing_identifier(self):
        with pytest.raises(InvalidInputError, match="missing"):
            ndcg(RECOMMENDATIONS, frame([("u", None)], ["user", "item"]))

    def test_no_relevant_item_at_all(self):
        with pytest.raises(InvalidInputError, match="relevant"):
            ndcg(RECOMMENDATIONS, self.truth.assign(rank=0.0))


@pytest.mark.parametrize("k", [1, 5, 10, 50, None])
@pytest.mark.parametrize("gain", ["linear", "exponential"])
def test_agrees_with_the_reference_implementation(random_data, k, gain):
    recommendations, truth = random_data["recommendations"], random_data["ground_truth"]
    lists = lists_from_rows(recommendations.itertuples(index=False))
    relevance = {}
    for user, item, grade in truth.itertuples(index=False):
        relevance.setdefault(user, {})[item] = grade
    expected = reference_ndcg(lists, relevance, k, exponential=gain == "exponential")
    assert ndcg(recommendations, truth, k=k, gain=gain) == pytest.approx(expected, rel=1e-12)


class TestReviewRegressions:
    """Inputs that used to give a wrong value or an undocumented exception."""

    recommendations = frame([("u", "a", 1), ("v", "a", 1)], ["user", "item", "rank"])

    def test_relevance_too_large_for_the_exponential_gain(self):
        # 2 ** 1024 is not a finite float: the perfectly ranked user "u" used to be dropped.
        truth = frame([("u", "a", 1024.0), ("v", "b", 1.0)], ["user", "item", "rank"])
        with pytest.raises(InvalidInputError, match="exponential"):
            ndcg(self.recommendations, truth, gain="exponential")
        scores = ndcg(self.recommendations, truth, gain="linear", per_user=True)
        assert scores.to_dict() == {"u": 1.0, "v": 0.0}
        truth = truth.assign(rank=[1023.0, 1.0])
        scores = ndcg(self.recommendations, truth, gain="exponential", per_user=True)
        assert scores.to_dict() == {"u": 1.0, "v": 0.0}

    def test_two_items_of_a_user_cannot_share_a_rank(self):
        tied = frame([("u", "a", 1), ("u", "b", 1)], ["user", "item", "rank"])
        truth = frame([("u", "a")], ["user", "item"])
        for table in (tied, tied.iloc[::-1]):
            with pytest.raises(InvalidInputError, match="same rank"):
                ndcg(table, truth)

    def test_identifiers_of_different_types_are_an_error_not_a_zero(self):
        recommendations = frame([(1, 10, 1), (1, 20, 2)], ["user", "item", "rank"])
        truth = frame([(1, 20)], ["user", "item"])
        assert ndcg(recommendations, truth) == pytest.approx(1 / math.log2(3))
        with pytest.raises(InvalidInputError, match=r"integer.*string|string.*integer"):
            ndcg(recommendations, truth.astype(str))

    def test_a_user_without_hits_scores_positive_zero(self):
        # "w" has a relevant item but no recommendation: its DCG is an empty sum.
        truth = frame([("w", "a")], ["user", "item"])
        score = ndcg(self.recommendations, truth, per_user=True)["w"]
        assert score == 0.0
        assert not math.copysign(1.0, score) < 0

    def test_any_cutoff_beyond_the_lists_is_the_whole_list(self):
        truth = frame([("u", "a")], ["user", "item"])
        assert ndcg(self.recommendations, truth, k=2**70) == ndcg(self.recommendations, truth)

    def test_user_without_hits_scores_positive_zero(self):
        import numpy as np

        truth = frame([("u", "z")], ["user", "item"])
        scores = ndcg(self.recommendations, truth, per_user=True)
        assert scores["u"] == 0.0
        assert not np.signbit(scores["u"])

    @pytest.mark.parametrize(("gain", "relevance"), [("exponential", 1023.0), ("linear", 1e308)])
    def test_gains_too_large_to_add(self, gain, relevance):
        # Every gain is finite, but three of them overflow when added: user "u", whose
        # ranking is perfect, used to be dropped as if nothing were relevant for them.
        recommendations = frame(
            [("u", "a", 1), ("u", "b", 2), ("u", "c", 3), ("v", "a", 1)], ["user", "item", "rank"]
        )
        truth = frame(
            [("u", "a", relevance), ("u", "b", relevance), ("u", "c", relevance), ("v", "z", 1.0)],
            ["user", "item", "rank"],
        )
        with pytest.raises(InvalidInputError, match="too large to be added"):
            ndcg(recommendations, truth, gain=gain)

    def test_integer_ranks_too_large_to_tell_apart(self):
        # 2**53 and 2**53 + 1 are the same float: say so, instead of reporting a tie.
        recommendations = frame(
            [("u", "a", 2**53), ("u", "b", 2**53 + 1)], ["user", "item", "rank"]
        )
        with pytest.raises(InvalidInputError, match=r"rank.*2\^53"):
            ndcg(recommendations, frame([("u", "a")], ["user", "item"]))
