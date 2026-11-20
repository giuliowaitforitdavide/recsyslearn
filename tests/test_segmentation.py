"""Segmentations of items and users.

* cumulative share of interactions: Abdollahpouri et al. (2019, 2021);
* share of the population: Li et al. (2021), Abdollahpouri et al. (2019);
* popularity: Vargas & Castells (2011) and Zhou et al. (2010);
* categorical attribute: Ekstrand et al. (2018).
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import frame
from reference import by_cumulative_interactions as reference_cumulative
from reference import by_interaction_count as reference_count
from reference import popularity as reference_popularity

from recsyslearn.errors import InvalidInputError, MissingColumnsError
from recsyslearn.segmentation import (
    by_cumulative_interactions,
    by_feature,
    by_interaction_count,
    popularity,
)


def interactions_with(counts, by="item"):
    """One interaction per row, each from a different counterpart."""
    other = "user" if by == "item" else "item"
    rows = [(entity, f"{other}{n}") for entity, count in counts.items() for n in range(count)]
    return frame(rows, [by, other]).sample(frac=1, random_state=11).reset_index(drop=True)


def groups_of(result, by="item"):
    return dict(zip(result[by], result["group"], strict=True))


class TestByCumulativeInteractions:
    def test_short_head_accounts_for_its_share_of_interactions(self):
        interactions = interactions_with({"a": 50, "b": 30, "c": 10, "d": 5, "e": 5})
        result = by_cumulative_interactions(interactions, (0.8, 0.2))
        assert list(result.columns) == ["item", "group"]
        assert groups_of(result) == {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2}

    def test_item_crossing_the_threshold_is_in_the_head(self):
        # 50 + 25 = 75% < 80%: the head is the smallest set that reaches 80%.
        interactions = interactions_with({"a": 50, "b": 25, "c": 15, "d": 5, "e": 5})
        assert groups_of(by_cumulative_interactions(interactions)) == {
            "a": 1,
            "b": 1,
            "c": 1,
            "d": 2,
            "e": 2,
        }

    def test_head_mid_tail(self):
        counts = {"a": 30, "b": 20, "c": 20, "d": 10, "e": 10, "f": 5, "g": 5}
        result = by_cumulative_interactions(interactions_with(counts), (0.2, 0.6, 0.2))
        assert groups_of(result) == {"a": 1, "b": 2, "c": 2, "d": 2, "e": 3, "f": 3, "g": 3}

    def test_labels(self):
        interactions = interactions_with({"a": 50, "b": 30, "c": 10, "d": 5, "e": 5})
        result = by_cumulative_interactions(interactions, labels=("head", "tail"))
        assert groups_of(result) == {
            "a": "head",
            "b": "head",
            "c": "tail",
            "d": "tail",
            "e": "tail",
        }

    def test_ties_are_broken_by_identifier(self):
        interactions = interactions_with({"d": 1, "b": 1, "c": 1, "a": 1})
        result = by_cumulative_interactions(interactions, (0.5, 0.5))
        assert groups_of(result) == {"a": 1, "b": 1, "c": 2, "d": 2}

    def test_users(self):
        interactions = interactions_with({"u1": 6, "u2": 2, "u3": 2}, by="user")
        result = by_cumulative_interactions(interactions, (0.5, 0.5), by="user")
        assert list(result.columns) == ["user", "group"]
        assert groups_of(result, "user") == {"u1": 1, "u2": 2, "u3": 2}

    def test_repeated_interactions_count(self):
        interactions = frame([("u", "a")] * 8 + [("u", "b")] * 2, ["user", "item"])
        assert groups_of(by_cumulative_interactions(interactions, (0.8, 0.2))) == {"a": 1, "b": 2}

    def test_output_is_sorted_by_identifier(self):
        interactions = interactions_with({"b": 1, "c": 5, "a": 3})
        assert list(by_cumulative_interactions(interactions)["item"]) == ["a", "b", "c"]

    def test_integer_identifiers_keep_their_dtype(self):
        interactions = frame([(1, 10), (2, 10), (1, 20)], ["user", "item"])
        result = by_cumulative_interactions(interactions, (0.5, 0.5))
        assert result["item"].tolist() == [10, 20]
        assert result["group"].tolist() == [1, 2]


class TestByInteractionCount:
    def test_top_share_of_the_population(self):
        interactions = interactions_with({"a": 9, "b": 7, "c": 5, "d": 3, "e": 1})
        result = by_interaction_count(interactions, (0.2, 0.8))
        assert groups_of(result) == {"a": 1, "b": 2, "c": 2, "d": 2, "e": 2}

    def test_most_active_five_percent_of_users(self):
        # Li et al. 2021: ceil(0.05 * 10) = 1 advantaged user.
        counts = {f"u{n}": n for n in range(1, 11)}
        result = by_interaction_count(
            interactions_with(counts, by="user"),
            (0.05, 0.95),
            by="user",
            labels=("advantaged", "disadvantaged"),
        )
        groups = groups_of(result, "user")
        assert groups.pop("u10") == "advantaged"
        assert set(groups.values()) == {"disadvantaged"}

    def test_population_shares_round_up(self):
        counts = {"a": 70, "b": 60, "c": 50, "d": 40, "e": 30, "f": 20, "g": 10}
        result = by_interaction_count(interactions_with(counts), (0.2, 0.6, 0.2))
        assert groups_of(result) == {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2, "f": 2, "g": 3}

    def test_cumulative_boundaries_are_rounded_not_group_sizes(self):
        # n = 5 and 30/30/40: boundaries at ceil(1.5) = 2 and ceil(3.0) = 3, so the groups
        # have 2, 1 and 2 items (rounding each size up would give 2, 2 and 2).
        interactions = interactions_with({"a": 5, "b": 4, "c": 3, "d": 2, "e": 1})
        result = by_interaction_count(interactions, (0.3, 0.3, 0.4))
        assert groups_of(result) == {"a": 1, "b": 1, "c": 2, "d": 3, "e": 3}

    def test_differs_from_the_cumulative_segmentation(self):
        interactions = interactions_with({"a": 96, "b": 1, "c": 1, "d": 1, "e": 1})
        by_population = by_interaction_count(interactions, (0.4, 0.6))
        by_interactions = by_cumulative_interactions(interactions, (0.4, 0.6))
        assert groups_of(by_population) == {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2}
        assert groups_of(by_interactions) == {"a": 1, "b": 2, "c": 2, "d": 2, "e": 2}


@pytest.mark.parametrize("segment", [by_cumulative_interactions, by_interaction_count])
class TestProportionErrors:
    interactions = interactions_with({"a": 3, "b": 1})

    @pytest.mark.parametrize("proportions", [(0.5, 0.4), (1.2, -0.2), (0.0, 1.0), (), ("a", "b")])
    def test_invalid_proportions(self, segment, proportions):
        with pytest.raises(InvalidInputError, match="proportions"):
            segment(self.interactions, proportions)

    def test_labels_must_match_the_groups(self, segment):
        with pytest.raises(InvalidInputError, match="labels"):
            segment(self.interactions, (0.8, 0.2), labels=("head",))
        with pytest.raises(InvalidInputError, match="labels"):
            segment(self.interactions, (0.8, 0.2), labels=("same", "same"))

    def test_invalid_side(self, segment):
        with pytest.raises(InvalidInputError, match="by must be"):
            segment(self.interactions, by="session")

    def test_missing_columns(self, segment):
        with pytest.raises(MissingColumnsError, match="item"):
            segment(self.interactions[["user"]])

    def test_empty_interactions(self, segment):
        result = segment(self.interactions.iloc[:0])
        assert list(result.columns) == ["item", "group"]
        assert result.empty


class TestPopularity:
    interactions = frame(
        [
            ("u1", "a"),
            ("u1", "a"),
            ("u2", "a"),
            ("u3", "a"),
            ("u1", "b"),
            ("u2", "b"),
            ("u4", "c"),
            ("u4", "c"),
        ],
        ["user", "item"],
    )

    def test_share_of_interactions(self):
        result = popularity(self.interactions)
        assert list(result.columns) == ["item", "popularity"]
        assert dict(zip(result["item"], result["popularity"], strict=True)) == {
            "a": 0.5,
            "b": 0.25,
            "c": 0.25,
        }
        assert result["popularity"].sum() == pytest.approx(1.0)

    def test_share_of_users(self):
        # |U_a| = 3, |U_b| = 2, |U_c| = 1 out of |U| = 4: repeated pairs count once.
        result = popularity(self.interactions, relative_to="population")
        assert dict(zip(result["item"], result["popularity"], strict=True)) == {
            "a": 0.75,
            "b": 0.5,
            "c": 0.25,
        }

    def test_user_activity(self):
        result = popularity(self.interactions, by="user")
        assert dict(zip(result["user"], result["popularity"], strict=True)) == {
            "u1": 0.375,
            "u2": 0.25,
            "u3": 0.125,
            "u4": 0.25,
        }
        result = popularity(self.interactions, by="user", relative_to="population")
        # Fraction of the three items each user interacted with.
        assert dict(zip(result["user"], result["popularity"], strict=True)) == pytest.approx(
            {"u1": 2 / 3, "u2": 2 / 3, "u3": 1 / 3, "u4": 1 / 3}
        )

    def test_errors(self):
        with pytest.raises(InvalidInputError, match="relative_to"):
            popularity(self.interactions, relative_to="ratings")
        with pytest.raises(MissingColumnsError, match="user"):
            popularity(self.interactions[["item"]])

    def test_empty_interactions(self):
        assert popularity(self.interactions.iloc[:0]).empty


class TestByFeature:
    users = frame(
        [("u1", "F", 25), ("u2", "M", 31), ("u3", None, 25), ("u4", "F", 47)],
        ["user", "gender", "age"],
    )

    def test_groups_are_the_attribute_values(self):
        result = by_feature(self.users, "age")
        assert list(result.columns) == ["user", "group"]
        assert groups_of(result, "user") == {"u1": 25, "u2": 31, "u3": 25, "u4": 47}

    def test_missing_values_form_their_own_group(self):
        result = by_feature(self.users, "gender")
        assert groups_of(result, "user") == {"u1": "F", "u2": "M", "u3": "unknown", "u4": "F"}

    def test_custom_label_for_missing_values(self):
        result = by_feature(self.users, "gender", missing="n/a")
        assert groups_of(result, "user")["u3"] == "n/a"

    def test_label_for_missing_values_must_be_new(self):
        with pytest.raises(InvalidInputError, match="missing"):
            by_feature(self.users, "gender", missing="F")

    def test_items(self):
        items = frame([("i1", "rock"), ("i2", "jazz")], ["item", "genre"])
        assert groups_of(by_feature(items, "genre")) == {"i1": "rock", "i2": "jazz"}

    def test_input_is_not_modified(self):
        before = self.users.copy()
        by_feature(self.users, "gender")
        pd.testing.assert_frame_equal(self.users, before)

    def test_errors(self):
        with pytest.raises(MissingColumnsError, match="country"):
            by_feature(self.users, "country")
        with pytest.raises(InvalidInputError, match="either"):
            by_feature(frame([("x", 1)], ["session", "length"]), "length")
        with pytest.raises(InvalidInputError, match="more than once"):
            by_feature(pd.concat([self.users, self.users]), "age")


def test_segmentation_feeds_the_fairness_metrics():
    from recsyslearn.fairness import kl_divergence

    interactions = interactions_with({"a": 50, "b": 30, "c": 10, "d": 5, "e": 5})
    groups = by_cumulative_interactions(interactions, labels=("head", "tail"))
    recommendations = frame([("u", "a", 1), ("u", "c", 2)], ["user", "item", "rank"])
    assert kl_divergence(recommendations, groups, exposure="flat") == pytest.approx(0.0)


PROPORTIONS = [(0.8, 0.2), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7), (0.05, 0.95), (1.0,)]


@pytest.mark.parametrize("by", ["item", "user"])
@pytest.mark.parametrize("proportions", PROPORTIONS)
def test_segmentations_agree_with_the_reference_implementation(random_data, by, proportions):
    interactions = random_data["interactions"]
    entities = interactions[by].tolist()

    actual = by_cumulative_interactions(interactions, proportions, by=by)
    assert groups_of(actual, by) == reference_cumulative(entities, proportions)

    actual = by_interaction_count(interactions, proportions, by=by)
    assert groups_of(actual, by) == reference_count(entities, proportions)


@pytest.mark.parametrize("by", ["item", "user"])
@pytest.mark.parametrize("relative_to", ["interactions", "population"])
def test_popularity_agrees_with_the_reference_implementation(random_data, by, relative_to):
    interactions = random_data["interactions"]
    other = "user" if by == "item" else "item"
    pairs = list(zip(interactions[by], interactions[other], strict=True))
    expected = reference_popularity(pairs, relative_to == "population")

    actual = popularity(interactions, by=by, relative_to=relative_to)
    assert dict(zip(actual[by], actual["popularity"], strict=True)) == pytest.approx(expected)


class TestReviewRegressions:
    """Inputs that used to give a wrong value or an undocumented exception."""

    def test_labels_must_be_a_collection(self):
        interactions = interactions_with({"a": 3, "b": 1})
        with pytest.raises(InvalidInputError, match="labels"):
            by_cumulative_interactions(interactions, (0.5, 0.5), labels="ht")

    @pytest.mark.parametrize("labels", [["head", None], ["head", float("nan")], [pd.NA, "tail"]])
    def test_labels_must_not_be_missing(self, labels):
        # A missing label would give a table that the fairness metrics refuse.
        interactions = interactions_with({"a": 3, "b": 1})
        with pytest.raises(InvalidInputError, match=r"labels.*missing"):
            by_cumulative_interactions(interactions, (0.5, 0.5), labels=labels)

    @pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
    def test_label_for_missing_values_must_not_be_missing(self, missing):
        features = frame([("u1", "F"), ("u2", None)], ["user", "gender"])
        with pytest.raises(InvalidInputError, match=r"missing.*label"):
            by_feature(features, "gender", missing=missing)

    def test_feature_table_with_a_missing_identifier(self):
        features = frame([("u1", "F"), (None, "M")], ["user", "gender"])
        with pytest.raises(InvalidInputError, match="identifiers must not be missing"):
            by_feature(features, "gender")


class TestEdgeCases:
    @pytest.mark.parametrize(
        "segment", [by_cumulative_interactions, by_interaction_count, popularity]
    )
    @pytest.mark.parametrize("column", ["user", "item"])
    def test_missing_identifiers(self, segment, column):
        interactions = frame([("u1", "a"), ("u2", "b")], ["user", "item"])
        interactions.loc[1, column] = None
        with pytest.raises(InvalidInputError, match="identifiers must not be missing"):
            segment(interactions)

    def test_popularity_is_sorted_by_identifier(self):
        interactions = frame([("u1", "c"), ("u2", "a"), ("u3", "b")], ["user", "item"])
        assert list(popularity(interactions)["item"]) == ["a", "b", "c"]

    def test_by_feature_keeps_the_order_of_the_rows(self):
        features = frame([("u3", "F"), ("u1", "M"), ("u2", "F")], ["user", "gender"])
        result = by_feature(features, "gender")
        assert list(result["user"]) == ["u3", "u1", "u2"]
        assert list(result["group"]) == ["F", "M", "F"]

    def test_rank_column_is_ignored(self):
        interactions = frame(
            [("u1", "a", 5), ("u2", "a", 1), ("u1", "b", 3)], ["user", "item", "rank"]
        )
        without = interactions.drop(columns="rank")
        for segment in (by_cumulative_interactions, by_interaction_count, popularity):
            pd.testing.assert_frame_equal(segment(interactions), segment(without))

    def test_proportions_must_sum_to_one_within_a_billionth(self):
        interactions = interactions_with({"a": 3, "b": 1})
        assert groups_of(by_interaction_count(interactions, (0.5, 0.5 + 5e-10))) == {"a": 1, "b": 2}
        with pytest.raises(InvalidInputError, match="sum to 1"):
            by_interaction_count(interactions, (0.5, 0.5 + 5e-9))

    def test_a_later_group_can_be_empty(self):
        # n = 2 and 30/30/40: boundaries at ceil(0.6) = 1 and ceil(1.2) = 2.
        interactions = interactions_with({"a": 3, "b": 1})
        result = by_interaction_count(interactions, (0.3, 0.3, 0.4), labels=("x", "y", "z"))
        assert groups_of(result) == {"a": "x", "b": "y"}

    def test_a_tiny_share_still_takes_the_first_entity(self):
        interactions = interactions_with({"a": 3, "b": 1, "c": 1})
        for segment in (by_cumulative_interactions, by_interaction_count):
            result = segment(interactions, (1e-16, 1 - 1e-16))
            assert groups_of(result) == {"a": 1, "b": 2, "c": 2}

    def test_missing_label_may_be_a_value_when_nothing_is_missing(self):
        features = frame([("u1", "F"), ("u2", "M")], ["user", "gender"])
        assert groups_of(by_feature(features, "gender", missing="F"), "user") == {
            "u1": "F",
            "u2": "M",
        }

    @pytest.mark.parametrize("labels", [[(1, 2), (3, 4)], [{}, {1: 2}], [[1], [2]]])
    def test_labels_must_be_scalars(self, labels):
        interactions = interactions_with({"a": 3, "b": 1})
        with pytest.raises(InvalidInputError, match=r"labels.*scalar"):
            by_interaction_count(interactions, (0.5, 0.5), labels=labels)

    @pytest.mark.parametrize("missing", [[1, 2], [1, 2, 3], (1, 2), {"a": 1}])
    def test_label_for_missing_values_must_be_a_scalar(self, missing):
        features = frame([("u1", "m"), ("u2", None), ("u3", "f")], ["user", "gender"])
        with pytest.raises(InvalidInputError, match=r"missing.*scalar"):
            by_feature(features, "gender", missing=missing)
