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

    def numbered(table):
        columns = {"user": users, "item": items}
        return table.assign(
            **{name: table[name].map(codes) for name, codes in columns.items() if name in table}
        )

    recommendations, truth, groups = numbered(RECOMMENDATIONS), numbered(TRUTH), numbered(GROUPS)
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


class TestReviewOfTheWholeChange:
    """Inputs found by the review of the pull request to be misread or badly refused."""

    def test_large_integers_of_different_types_are_refused(self):
        # Combined, int64 and uint64 become float64, where 2**60 and 2**60 + 1 are the
        # same number: the two users would be merged into one.
        import numpy as np

        from recsyslearn.errors import InvalidInputError

        users = [2**60, 2**60 + 1]
        recommendations = pd.DataFrame({"user": np.array(users, dtype="uint64"), "item": [1, 2]})
        truth = pd.DataFrame({"user": np.array(users, dtype="int64"), "item": [1, 3]})
        with pytest.raises(InvalidInputError, match=r"user identifiers.*cannot be told apart"):
            accuracy.ndcg(recommendations, truth)
        # Below 2**53 the two types hold the same numbers exactly.
        small = recommendations.assign(user=np.array([5, 6], dtype="uint64"))
        assert accuracy.ndcg(small, truth.assign(user=[5, 6])) == pytest.approx(0.5)

    def test_unhashable_identifiers_are_refused_everywhere(self):
        from recsyslearn.errors import InvalidInputError

        groups = pd.DataFrame({"item": [["x"], ["y"], ["z"]], "group": ["a", "b", "b"]})
        with pytest.raises(InvalidInputError, match="item identifiers must be hashable"):
            fairness.group_distribution(RECOMMENDATIONS, groups)
        lists = pd.DataFrame({"user": ["a", "a"], "item": [["x"], ["y"]]})
        with pytest.raises(InvalidInputError, match="item identifiers must be hashable"):
            beyond_accuracy.coverage(lists, ["x", "y"])
        features = pd.DataFrame({"item": [["x"], ["y"]], "genre": ["a", "b"]})
        with pytest.raises(InvalidInputError, match="item identifiers must be hashable"):
            segmentation.by_feature(features, "genre")

    def test_by_feature_reads_its_identifier_column_once(self):
        from recsyslearn.errors import InvalidInputError

        features = pd.DataFrame([[1, 1, "a"], [2, 2, "b"]], columns=["item", "item", "genre"])
        with pytest.raises(InvalidInputError, match="more than one column named 'item'"):
            segmentation.by_feature(features, "genre")
        with pytest.raises(InvalidInputError, match="column must be the name of a column"):
            segmentation.by_feature(features, ["genre"])

    def test_by_feature_can_group_by_the_other_identifier(self):
        # `column` can itself be `user` or `item`: the other one is then what is grouped.
        features = frame([("a", "x"), ("b", "y")], ["user", "item"])
        grouped = segmentation.by_feature(features, "item")
        assert grouped.to_dict("list") == {"user": ["a", "b"], "group": ["x", "y"]}

    def test_labels_must_have_an_order(self):
        from recsyslearn.errors import InvalidInputError

        with pytest.raises(InvalidInputError, match="labels must be an ordered"):
            segmentation.by_cumulative_interactions(TRUTH, (0.5, 0.5), labels={"head", "tail"})

    def test_datetimes_with_and_without_a_time_zone_do_not_match(self):
        from recsyslearn.errors import InvalidInputError

        days = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"])
        recommendations = RECOMMENDATIONS.assign(user=days)
        truth = TRUTH.assign(user=days[[0, 2]].tz_localize("UTC"))
        with pytest.raises(InvalidInputError, match=r"user identifiers.*different types"):
            accuracy.ndcg(recommendations, truth)

    def test_errors_survive_pickling(self):
        import copy
        import pickle

        from recsyslearn.errors import InvalidInputError, MissingColumnsError

        error = MissingColumnsError("recommendations", ["user", "item"])
        for clone in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
            assert (clone.name, clone.missing) == ("recommendations", ("user", "item"))
            assert str(clone) == str(error)
        assert str(pickle.loads(pickle.dumps(InvalidInputError("no")))) == "no"

    @pytest.mark.parametrize(
        "target",
        [{"head": "0.5", "tail": "0.5"}, {"head": True, "tail": False}],
        ids=["strings", "booleans"],
    )
    def test_target_probabilities_must_be_numbers(self, target):
        from recsyslearn.errors import InvalidInputError

        with pytest.raises(InvalidInputError, match="probabilities of target must be numeric"):
            fairness.kl_divergence(RECOMMENDATIONS, GROUPS, target)

    @pytest.mark.parametrize("proportions", [["0.5", "0.5"], [True], [0.5, None]])
    def test_proportions_must_be_numbers(self, proportions):
        from recsyslearn.errors import InvalidInputError

        with pytest.raises(InvalidInputError, match="proportions must be a sequence of numbers"):
            segmentation.by_interaction_count(TRUTH, proportions)

    def test_a_catalog_of_tuples_given_as_a_multi_index(self):
        lists = pd.DataFrame({"user": ["a", "a"], "item": [(1, 2), (3, 4)]})
        catalog = pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)])
        assert beyond_accuracy.coverage(lists, catalog) == pytest.approx(2 / 3)

    def test_columns_with_several_levels_are_refused(self):
        from recsyslearn.errors import InvalidInputError

        nested = RECOMMENDATIONS.set_axis(
            pd.MultiIndex.from_product([["user", "item", "rank"], ["a"]]), axis=1
        )
        with pytest.raises(InvalidInputError, match=r"recommendations.*single level"):
            accuracy.ndcg(nested, TRUTH)
        with pytest.raises(InvalidInputError, match=r"interactions.*single level"):
            segmentation.popularity(nested)

    def test_a_missing_ground_truth_is_named(self):
        with pytest.raises(TypeError, match="ground_truth must be a pandas DataFrame"):
            accuracy.ndcg(RECOMMENDATIONS, None)

    @pytest.mark.parametrize("value", [["linear"], pd.NA, None, 1])
    def test_options_must_be_strings(self, value):
        from recsyslearn.errors import InvalidInputError

        with pytest.raises(InvalidInputError, match="gain must be 'linear' or 'exponential'"):
            accuracy.ndcg(RECOMMENDATIONS, TRUTH, gain=value)

    def test_errors_about_groups_name_the_argument(self):
        from recsyslearn.errors import InvalidInputError

        short = GROUPS.head(1)
        with pytest.raises(InvalidInputError, match="have no group in groups "):
            fairness.entropy(RECOMMENDATIONS, short)
        with pytest.raises(InvalidInputError, match="have no group in item_groups "):
            fairness.mutual_information(RECOMMENDATIONS, item_groups=short)

    def test_the_largest_exact_integer_is_accepted(self):
        # 2**53 is still told apart from its neighbours below: only what lies beyond is refused.
        ranked = RECOMMENDATIONS.assign(rank=[1, 2**53, 1, 2**53])
        assert accuracy.ndcg(ranked, TRUTH) == accuracy.ndcg(RECOMMENDATIONS, TRUTH)
        negative = RECOMMENDATIONS.assign(rank=[-(2**53), 2, -(2**53), 2])
        assert accuracy.ndcg(negative, TRUTH) == accuracy.ndcg(RECOMMENDATIONS, TRUTH)


def test_results_do_not_depend_on_the_number_of_threads():
    """The same tied, grouped data give bit-identical results with one thread and with four."""
    import os
    import subprocess
    import sys

    script = """
import numpy as np, pandas as pd
from recsyslearn import accuracy, beyond_accuracy, fairness
rng = np.random.default_rng(7)
n_users, n_items, length = 400, 60, 12
items = np.argsort(rng.random((n_users, n_items)), axis=1)[:, :length].ravel()
users = np.repeat(np.arange(n_users), length)
lists = pd.DataFrame({"user": users, "item": items, "score": rng.integers(0, 4, len(users))})
truth = lists.sample(frac=0.2, random_state=1)[["user", "item"]].assign(rating=2.5)
groups = pd.DataFrame({"item": np.arange(n_items), "group": np.arange(n_items) % 3})
values = [
    accuracy.ndcg(lists, truth, k=5),
    beyond_accuracy.novelty(lists, truth, k=5),
    fairness.entropy(lists, groups, k=5, ground_truth=truth),
    fairness.mutual_information(lists, item_groups=groups, k=5),
    *accuracy.ndcg(lists, truth, per_user=True),
]
print(" ".join(float(value).hex() for value in values))
"""

    def run(threads):
        environment = {**os.environ, "RAYON_NUM_THREADS": threads}
        done = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout

    single = run("1")
    assert len(single.split()) > 100
    assert run("4") == single
