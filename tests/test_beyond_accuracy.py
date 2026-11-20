"""Catalog coverage (Ge et al. 2010) and novelty as mean self-information
(Zhou et al. 2010; Schedl et al. 2018, Eq. 12)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from conftest import frame
from reference import coverage as reference_coverage
from reference import lists_from_rows
from reference import novelty as reference_novelty

from recsyslearn.beyond_accuracy import coverage, novelty
from recsyslearn.errors import InvalidInputError, MissingColumnsError

# Five users, catalog {A, B, C, D}, lists of length two.
LISTS = {"u1": "AB", "u2": "AB", "u3": "AC", "u4": "BA", "u5": "AB"}
RECOMMENDATIONS = frame(
    [(user, item, rank) for user, items in LISTS.items() for rank, item in enumerate(items, 1)],
    ["user", "item", "rank"],
)
# Training interactions with user-degrees k_A = 4, k_B = 2, k_C = 1, k_D = 1.
INTERACTIONS = frame(
    [
        ("u1", "A"),
        ("u2", "A"),
        ("u3", "A"),
        ("u4", "A"),
        ("u1", "B"),
        ("u5", "B"),
        ("u2", "C"),
        ("u3", "D"),
    ],
    ["user", "item"],
)


class TestCoverage:
    def test_share_of_the_catalog_that_is_recommended(self):
        assert coverage(RECOMMENDATIONS, list("ABCD")) == 0.75

    def test_cutoff(self):
        assert coverage(RECOMMENDATIONS, list("ABCD"), k=1) == 0.5

    @pytest.mark.parametrize(
        "catalog",
        [
            pd.Series(list("ABCD")),
            INTERACTIONS,
            frame(list("ABCD"), ["item"]),
            {"A", "B", "C", "D"},
            list("ABCDDDA"),
        ],
        ids=["series", "interactions", "frame", "set", "with-duplicates"],
    )
    def test_catalog_can_be_any_collection_of_items(self, catalog):
        assert coverage(RECOMMENDATIONS, catalog) == 0.75

    def test_full_coverage(self):
        assert coverage(RECOMMENDATIONS, list("ABC")) == 1.0

    def test_item_outside_the_catalog(self):
        with pytest.raises(InvalidInputError, match="catalog"):
            coverage(RECOMMENDATIONS, list("AB"))

    def test_empty_catalog(self):
        with pytest.raises(InvalidInputError, match="catalog"):
            coverage(RECOMMENDATIONS.iloc[:0], [])

    def test_missing_columns(self):
        with pytest.raises(MissingColumnsError, match="item"):
            coverage(RECOMMENDATIONS[["user"]], list("ABCD"))
        with pytest.raises(MissingColumnsError, match="item"):
            coverage(RECOMMENDATIONS, frame(list("ABCD"), ["movie"]))

    @pytest.mark.parametrize("k", [1, 2, 7, None])
    def test_agrees_with_the_reference_implementation(self, random_data, k):
        recommendations = random_data["recommendations"]
        catalog = [f"i{item}" for item in range(80)]
        lists = lists_from_rows(recommendations.itertuples(index=False))
        assert coverage(recommendations, catalog, k=k) == pytest.approx(
            reference_coverage(lists, catalog, k)
        )


class TestNovelty:
    def test_mean_self_information(self):
        a, b, c = math.log2(5 / 4), math.log2(5 / 2), math.log2(5 / 1)
        expected = (4 * (a + b) / 2 + (a + c) / 2) / 5
        assert novelty(RECOMMENDATIONS, INTERACTIONS) == pytest.approx(expected)
        assert novelty(RECOMMENDATIONS, INTERACTIONS) == pytest.approx(0.921928, abs=1e-6)

    def test_per_user(self):
        scores = novelty(RECOMMENDATIONS, INTERACTIONS, per_user=True)
        assert isinstance(scores, pd.Series)
        assert scores.index.name == "user"
        assert scores.name == "novelty"
        # u1 is recommended A and B, u3 A and C; the user-degrees are 4, 2 and 1 out of 5.
        a, b, c = math.log2(5 / 4), math.log2(5 / 2), math.log2(5 / 1)
        assert scores["u1"] == pytest.approx((a + b) / 2)
        assert scores["u3"] == pytest.approx((a + c) / 2)
        assert scores["u1"] == pytest.approx(0.821928, abs=1e-6)
        assert scores["u3"] == pytest.approx(1.321928, abs=1e-6)
        assert scores.mean() == pytest.approx(novelty(RECOMMENDATIONS, INTERACTIONS))

    def test_cutoff(self):
        scores = novelty(RECOMMENDATIONS, INTERACTIONS, k=1, per_user=True)
        assert scores["u1"] == pytest.approx(math.log2(5 / 4))
        assert scores["u4"] == pytest.approx(math.log2(5 / 2))

    def test_popularity_counts_users_not_interactions(self):
        repeated = pd.concat([INTERACTIONS, INTERACTIONS.iloc[:3]], ignore_index=True)
        assert novelty(RECOMMENDATIONS, repeated) == pytest.approx(
            novelty(RECOMMENDATIONS, INTERACTIONS)
        )

    def test_explicit_feedback_values_are_ignored(self):
        rated = INTERACTIONS.assign(rank=4.0)
        assert novelty(RECOMMENDATIONS, rated) == pytest.approx(
            novelty(RECOMMENDATIONS, INTERACTIONS)
        )

    def test_item_known_by_everyone_is_not_novel(self):
        interactions = frame([("u1", "A"), ("u2", "A")], ["user", "item"])
        recommendations = frame([("u1", "A")], ["user", "item"])
        assert novelty(recommendations, interactions) == 0.0

    def test_unseen_item_is_as_novel_as_an_item_with_one_user(self):
        recommendations = frame([("u1", "never-seen")], ["user", "item"])
        assert novelty(recommendations, INTERACTIONS) == pytest.approx(math.log2(5))

    def test_users_are_those_of_the_interactions(self):
        # A recommended user who is not in the interactions does not change |U|.
        recommendations = frame([("stranger", "B")], ["user", "item"])
        assert novelty(recommendations, INTERACTIONS) == pytest.approx(math.log2(5 / 2))

    def test_errors(self):
        with pytest.raises(MissingColumnsError, match="user"):
            novelty(RECOMMENDATIONS, INTERACTIONS[["item"]])
        with pytest.raises(InvalidInputError, match="k must be a positive integer"):
            novelty(RECOMMENDATIONS, INTERACTIONS, k=0)
        with pytest.raises(InvalidInputError, match="no recommendations"):
            novelty(RECOMMENDATIONS.iloc[:0], INTERACTIONS)
        with pytest.raises(InvalidInputError, match="no users"):
            novelty(RECOMMENDATIONS, INTERACTIONS.iloc[:0])

    @pytest.mark.parametrize("k", [1, 5, 12, None])
    def test_agrees_with_the_reference_implementation(self, random_data, k):
        recommendations, interactions = random_data["recommendations"], random_data["interactions"]
        lists = lists_from_rows(recommendations.itertuples(index=False))
        pairs = list(interactions.itertuples(index=False, name=None))
        expected = reference_novelty(lists, pairs, k)
        assert novelty(recommendations, interactions, k=k) == pytest.approx(expected, rel=1e-12)


class TestReviewRegressions:
    """Inputs that used to give a wrong value or an undocumented exception."""

    def test_missing_identifier_in_the_catalog(self):
        with pytest.raises(InvalidInputError, match=r"catalog.*missing"):
            coverage(RECOMMENDATIONS, ["A", "B", "C", "D", None])

    def test_catalog_must_be_a_flat_collection(self):
        import numpy as np

        with pytest.raises(InvalidInputError, match="catalog"):
            coverage(RECOMMENDATIONS, np.array([["A", "B"], ["C", "D"]]))
        with pytest.raises(TypeError, match="catalog"):
            coverage(RECOMMENDATIONS, None)

    def test_item_identifiers_of_different_types_are_an_error(self):
        interactions = frame([(1, 10), (2, 10), (2, 20)], ["user", "item"])
        recommendations = frame([(1, 20)], ["user", "item"])
        assert novelty(recommendations, interactions) == pytest.approx(1.0)
        with pytest.raises(InvalidInputError, match=r"integer.*string|string.*integer"):
            novelty(recommendations, interactions.astype({"item": str}))


class TestEdgeCases:
    def test_cutoff_beyond_the_lists_adds_nothing(self):
        catalog = ["A", "B", "C", "D"]
        # Three of the four items of the catalog are recommended.
        assert coverage(RECOMMENDATIONS, catalog, k=100) == pytest.approx(3 / 4)
        assert novelty(RECOMMENDATIONS, INTERACTIONS, k=100) == pytest.approx(
            novelty(RECOMMENDATIONS, INTERACTIONS)
        )


class TestCatalogAsAList:
    recommendations = frame([("u", ("a", 1)), ("u", ("b", 2))], ["user", "item"])

    def test_identifiers_can_be_tuples(self):
        # Two of the three items of the catalog are recommended.
        catalog = [("a", 1), ("b", 2), ("c", 3)]
        assert coverage(self.recommendations, catalog) == pytest.approx(2 / 3)
        assert coverage(self.recommendations, pd.Series(catalog)) == pytest.approx(2 / 3)

    @pytest.mark.parametrize("catalog", [[["A", "B"], ["C"]], [["A", "B"], ["C", "D"]], [{}, {}]])
    def test_items_must_be_hashable(self, catalog):
        with pytest.raises(InvalidInputError, match="catalog"):
            coverage(RECOMMENDATIONS, catalog)

    def test_a_string_is_not_a_catalog(self):
        with pytest.raises(TypeError, match="catalog must be a collection"):
            coverage(RECOMMENDATIONS, "ABCD")

    def test_nothing_recommended_covers_nothing(self):
        assert coverage(RECOMMENDATIONS.iloc[:0], ["A", "B", "C", "D"]) == 0.0
