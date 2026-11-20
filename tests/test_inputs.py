"""The results do not depend on how the identifiers and the ranks are stored."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import frame

from recsyslearn import accuracy, beyond_accuracy, fairness, segmentation

RECOMMENDATIONS = frame(
    [("a", "x", 1), ("a", "y", 2), ("b", "x", 1), ("b", "z", 2)], ["user", "item", "rank"]
)
TRUTH = frame([("a", "y"), ("b", "z")], ["user", "item"])
GROUPS = frame([("x", "head"), ("y", "tail"), ("z", "tail")], ["item", "group"])
#: The same lists, ordered by score instead of rank.
SCORED = RECOMMENDATIONS.drop(columns="rank").assign(score=[0.9, 0.4, 0.9, 0.4])


def results(recommendations, truth, groups):
    return [
        accuracy.ndcg(recommendations, truth),
        beyond_accuracy.coverage(recommendations, groups),
        beyond_accuracy.novelty(recommendations, truth),
        fairness.entropy(recommendations, groups),
        fairness.mutual_information(recommendations, item_groups=groups),
        *segmentation.popularity(truth)["popularity"],
    ]


EXPECTED = results(RECOMMENDATIONS, TRUTH, GROUPS)


@pytest.mark.parametrize("dtype", ["category", "string", "object"])
def test_identifier_dtype(dtype):
    def convert(table):
        columns = [column for column in ("user", "item") if column in table]
        return table.astype(dict.fromkeys(columns, dtype))

    actual = results(convert(RECOMMENDATIONS), convert(TRUTH), convert(GROUPS))
    assert actual == pytest.approx(EXPECTED)


def test_integer_identifiers():
    users, items = {"a": 7, "b": 3}, {"x": 30, "y": 10, "z": 20}
    recommendations = RECOMMENDATIONS.replace({"user": users, "item": items})
    truth = TRUTH.replace({"user": users, "item": items})
    groups = GROUPS.replace({"item": items})
    assert results(recommendations, truth, groups) == pytest.approx(EXPECTED)


def test_index_is_ignored():
    recommendations = RECOMMENDATIONS.set_index(pd.Index([40, 30, 20, 10]))
    assert results(recommendations, TRUTH, GROUPS) == pytest.approx(EXPECTED)
    # Duplicated and unordered labels on the other tables as well.
    truth = TRUTH.set_index(pd.Index([5, 5]))
    groups = GROUPS.set_index(pd.Index([2, 0, 2]))
    assert results(recommendations, truth, groups) == pytest.approx(EXPECTED)
    features = frame([("a", "f"), ("b", "m")], ["user", "gender"])
    by_index = segmentation.by_feature(features.set_index(pd.Index([9, 9])), "gender")
    assert list(by_index["group"]) == ["f", "m"]


def test_results_do_not_depend_on_the_order_of_the_rows(random_data):
    """Bit for bit: sums are made in the order of the identifiers, not of the rows."""

    def compute(tables):
        recommendations, truth = tables["recommendations"], tables["ground_truth"]
        users, items = tables["user_groups"], tables["item_groups"]
        return [
            accuracy.ndcg(recommendations, truth),
            beyond_accuracy.novelty(recommendations, tables["interactions"]),
            *fairness.group_distribution(recommendations, items),
            fairness.entropy(recommendations, users, ground_truth=truth),
            fairness.kl_divergence(recommendations, items),
            fairness.mutual_information(recommendations, user_groups=users, item_groups=items),
            fairness.mutual_information(recommendations, item_groups=items),
            fairness.mutual_information(recommendations, user_groups=users, ground_truth=truth),
        ]

    expected = compute(random_data)
    for seed in range(5):
        shuffled = {
            name: table.sample(frac=1, random_state=seed) for name, table in random_data.items()
        }
        assert compute(shuffled) == expected


def test_results_with_tied_scores_do_not_depend_on_the_order_of_the_rows(tied_data):
    def compute(tables):
        recommendations, items = tables["recommendations"], tables["item_groups"]
        return [
            accuracy.ndcg(recommendations, tables["ground_truth"], k=3),
            beyond_accuracy.novelty(recommendations, tables["interactions"], k=3),
            *fairness.group_distribution(recommendations, items, k=3),
            fairness.mutual_information(recommendations, item_groups=items, k=3),
        ]

    expected = compute(tied_data)
    for seed in range(5):
        shuffled = {
            name: table.sample(frac=1, random_state=seed) for name, table in tied_data.items()
        }
        assert compute(shuffled) == expected


def test_scores_of_any_numeric_type():
    scores = {"x": 3, "y": 2, "z": 1}
    scored = RECOMMENDATIONS.assign(score=RECOMMENDATIONS["item"].map(scores)).drop(columns="rank")
    expected = accuracy.ndcg(RECOMMENDATIONS, TRUTH)
    for dtype in ("int8", "uint16", "Int64", "float32", "float64", "Float64"):
        assert accuracy.ndcg(scored.astype({"score": dtype}), TRUTH) == expected


@pytest.mark.parametrize(
    "columns",
    [["user", "item", "rank", "rank"], ["user", "user", "item"], ["user", "item", "item"]],
)
def test_duplicated_column_names_are_refused(columns):
    from recsyslearn.errors import InvalidInputError

    table = pd.DataFrame([[1] * len(columns), [2] * len(columns)], columns=columns)
    with pytest.raises(InvalidInputError, match="more than one column named"):
        accuracy.ndcg(table, TRUTH)
    groups = pd.DataFrame([["x", 1, 1], ["y", 2, 2]], columns=["item", "group", "group"])
    with pytest.raises(InvalidInputError, match="more than one column named 'group'"):
        fairness.entropy(RECOMMENDATIONS, groups)
    features = pd.DataFrame([["a", 1, 1], ["b", 2, 2]], columns=["user", "g", "g"])
    with pytest.raises(InvalidInputError, match="more than one column named 'g'"):
        segmentation.by_feature(features, "g")


def test_duplicated_columns_that_are_not_used_are_ignored():
    extra = pd.DataFrame({"group": ["g"] * 4, "other": [0] * 4})
    recommendations = pd.concat([RECOMMENDATIONS, extra, extra], axis=1)
    assert accuracy.ndcg(recommendations, TRUTH) == accuracy.ndcg(RECOMMENDATIONS, TRUTH)
    # The segmentations ignore `rating`, so two of them are no obstacle either.
    interactions = pd.concat(
        [TRUTH, pd.DataFrame({"rating": [1, 2]}), pd.DataFrame({"rating": [3, 4]})], axis=1
    )
    pd.testing.assert_frame_equal(
        segmentation.popularity(interactions), segmentation.popularity(TRUTH)
    )


def test_empty_table_of_another_dtype():
    # An empty frame has float columns: concatenating one with integer identifiers
    # warns on pandas 2.1, the oldest supported.
    empty = pd.DataFrame({"user": [], "item": []})
    truth = frame([(1, 10), (2, 20)], ["user", "item"])
    assert accuracy.ndcg(empty, truth) == 0.0


def test_identifiers_that_cannot_be_ordered_keep_the_order_of_first_appearance():
    # A number and a tuple cannot be compared, so the identifiers are not sorted.
    recommendations = frame([(("b", 2), "x"), (1, "x"), (("a", 1), "y")], ["user", "item"])
    truth = frame([(1, "x"), (("a", 1), "y")], ["user", "item"])
    scores = accuracy.ndcg(recommendations, truth, per_user=True)
    assert list(scores.index) == [1, ("a", 1)]
    assert list(scores) == [1.0, 1.0]


@pytest.mark.parametrize("dtype", ["uint8", "uint64", "UInt64", "int8"])
def test_ranks_of_any_integer_type(dtype):
    recommendations = RECOMMENDATIONS.astype({"rank": dtype})
    assert accuracy.ndcg(recommendations, TRUTH) == accuracy.ndcg(RECOMMENDATIONS, TRUTH)


@pytest.mark.parametrize("extreme", [2**53 + 1, -(2**53) - 1, -(2**63)])
def test_integer_ranks_beyond_the_exact_range_are_refused(extreme):
    from recsyslearn.errors import InvalidInputError

    recommendations = RECOMMENDATIONS.assign(rank=[extreme, 2, 1, 2])
    with pytest.raises(InvalidInputError, match=r"rank.*2\^53"):
        accuracy.ndcg(recommendations, TRUTH)


def test_integer_scores_beyond_the_exact_range_are_refused():
    from recsyslearn.errors import InvalidInputError

    scored = SCORED.assign(score=[2**53 + 1, 2, 1, 2])
    with pytest.raises(InvalidInputError, match=r"score.*2\^53"):
        accuracy.ndcg(scored, TRUTH)


NOT_NUMBERS = {
    "numeric-strings": ["1", "2", "1", "2"],
    "string-dtype": pd.array(["1", "2", "1", "2"], dtype="string"),
    "booleans": [True, False, True, False],
    # Two integers that float64 cannot tell apart, hidden in a column of objects.
    "integers-as-objects": pd.Series([2**53, 2**53 + 1, 1, 2], dtype=object),
    "dates": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
    "categories": pd.Categorical([1, 2, 1, 2]),
    "complex": [1 + 2j, 2 + 0j, 1 + 0j, 2 + 0j],
}


@pytest.mark.parametrize("cells", NOT_NUMBERS.values(), ids=NOT_NUMBERS.keys())
@pytest.mark.parametrize("column", ["rank", "score"])
def test_the_order_must_be_given_by_numbers(column, cells):
    from recsyslearn.errors import InvalidInputError

    recommendations = SCORED.drop(columns="score").assign(**{column: cells})
    with pytest.raises(InvalidInputError, match=f"'{column}' column of recommendations must be"):
        accuracy.ndcg(recommendations, TRUTH)


@pytest.mark.parametrize("cells", NOT_NUMBERS.values(), ids=NOT_NUMBERS.keys())
def test_ratings_must_be_numbers(cells):
    from recsyslearn.errors import InvalidInputError

    truth = RECOMMENDATIONS.drop(columns="rank").assign(rating=cells)
    with pytest.raises(InvalidInputError, match="'rating' column of ground_truth must be"):
        accuracy.ndcg(RECOMMENDATIONS, truth)


def test_a_score_next_to_a_rank_is_not_read():
    # Not read means not checked either: whatever the column holds, and however many
    # times it appears, the lists are those of the ranks.
    unreadable = RECOMMENDATIONS.assign(score=["high", None, "low", "low"])
    assert results(unreadable, TRUTH, GROUPS) == EXPECTED
    repeated = pd.concat([unreadable, unreadable[["score"]]], axis=1)
    assert results(repeated, TRUTH, GROUPS) == EXPECTED


def test_a_pair_recommended_twice_is_refused_with_scores():
    from recsyslearn.errors import InvalidInputError

    twice = pd.concat([SCORED, SCORED.head(1)], ignore_index=True)
    with pytest.raises(InvalidInputError, match="same \\(user, item\\) pair more than once"):
        accuracy.ndcg(twice, TRUTH)


def test_identifiers_must_be_hashable():
    from recsyslearn.errors import InvalidInputError

    recommendations = frame([("a", ["x"]), ("b", ["y"])], ["user", "item"])
    with pytest.raises(InvalidInputError, match="item identifiers must be hashable"):
        accuracy.ndcg(recommendations, TRUTH)


def test_more_identifiers_than_the_core_can_code(monkeypatch):
    from recsyslearn import _frames
    from recsyslearn.errors import InvalidInputError

    monkeypatch.setattr(_frames, "_MAX_CODES", 2)
    with pytest.raises(InvalidInputError, match="3 distinct item identifiers"):
        accuracy.ndcg(RECOMMENDATIONS, TRUTH)


@pytest.mark.parametrize(
    "ranks",
    [[0, 1, 0, 1], [10, 20, 5, 500], [-2.5, 0.0, 0.1, 0.2], pd.array([1, 2, 1, 2], dtype="Int32")],
    ids=["zero-based", "with-gaps", "floats", "nullable-integers"],
)
def test_only_the_order_of_the_ranks_matters(ranks):
    recommendations = RECOMMENDATIONS.assign(rank=ranks)
    assert results(recommendations, TRUTH, GROUPS) == pytest.approx(EXPECTED)


def test_no_function_modifies_its_arguments():
    features = frame([("a", "f"), ("b", None)], ["user", "gender"])
    tables = {
        "recommendations": RECOMMENDATIONS.sample(frac=1, random_state=1),
        # Tied scores, in an order that is not the one of the lists.
        "scored": SCORED.assign(score=[0.4, 0.4, 0.4, 0.9]),
        "truth": TRUTH.assign(rating=[2.0, 1.0]),
        "groups": GROUPS,
        "features": features,
    }
    before = {name: table.copy(deep=True) for name, table in tables.items()}
    recommendations, truth, groups = tables["recommendations"], tables["truth"], tables["groups"]

    results(recommendations, truth, groups)
    accuracy.ndcg(recommendations, truth, k=1, gain="exponential", per_user=True)
    beyond_accuracy.novelty(recommendations, truth, per_user=True)
    fairness.group_distribution(recommendations, groups, ground_truth=truth)
    fairness.kl_divergence(recommendations, groups, {"head": 0.5, "tail": 0.5}, exposure="flat")
    segmentation.by_cumulative_interactions(truth, labels=("head", "tail"))
    segmentation.by_interaction_count(truth, by="user")
    segmentation.popularity(truth, relative_to="population")
    segmentation.by_feature(features, "gender")
    scored = tables["scored"]
    accuracy.ndcg(scored, truth, k=1, per_user=True)
    beyond_accuracy.coverage(scored, groups)
    beyond_accuracy.novelty(scored, truth, k=1)
    fairness.group_distribution(scored, groups, k=1, ground_truth=truth)

    for name, table in tables.items():
        pd.testing.assert_frame_equal(table, before[name])


def test_rank_must_be_numeric():
    from recsyslearn.errors import InvalidInputError

    with pytest.raises(InvalidInputError, match="'rank' column of recommendations"):
        accuracy.ndcg(RECOMMENDATIONS.assign(rank=["first", "second", "first", "second"]), TRUTH)
    with pytest.raises(InvalidInputError, match="NaN rank"):
        accuracy.ndcg(RECOMMENDATIONS.assign(rank=[1.0, None, 1.0, 2.0]), TRUTH)
    with pytest.raises(InvalidInputError, match="'rating' column of ground_truth"):
        accuracy.ndcg(RECOMMENDATIONS, TRUTH.assign(rating=["good"] * len(TRUTH)))


def test_target_must_map_groups_to_probabilities():
    with pytest.raises(TypeError, match="target"):
        fairness.kl_divergence(RECOMMENDATIONS, GROUPS, [0.5, 0.5])


def test_missing_columns_error_says_which_table_and_columns():
    from recsyslearn.errors import InvalidInputError, MissingColumnsError, RecsyslearnError

    with pytest.raises(MissingColumnsError) as caught:
        accuracy.ndcg(RECOMMENDATIONS[["rank"]], TRUTH)
    assert caught.value.name == "recommendations"
    assert caught.value.missing == ("user", "item")
    for error in (MissingColumnsError, InvalidInputError):
        assert issubclass(error, RecsyslearnError)
        assert issubclass(error, ValueError)


def test_errors_do_not_carry_the_content_of_the_tables():
    from recsyslearn.errors import InvalidInputError

    secret = RECOMMENDATIONS.assign(rank=["secret-value", "2", "1", "2"])
    with pytest.raises(InvalidInputError) as caught:
        accuracy.ndcg(secret, TRUTH)
    assert caught.value.__cause__ is None
    assert "secret-value" not in str(caught.value)
