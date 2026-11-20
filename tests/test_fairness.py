"""Group fairness in the probabilistic framework of Amigó et al. (2023): the benefit
(exposure or effectiveness) is normalised over all the recommended pairs; inequity is
a KL divergence (Eq. 1), dependence a mutual information (Eq. 2)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from conftest import frame
from reference import entropy as reference_entropy
from reference import joint_distribution, lists_from_rows, marginal
from reference import kl_divergence as reference_kl
from reference import mutual_information as reference_mi

from recsyslearn.errors import InvalidInputError, MissingColumnsError
from recsyslearn.fairness import entropy, group_distribution, kl_divergence, mutual_information

# u1 is in user group A, u2 in B; i1 and i2 are in item group G1, i3 and i4 in G2.
LISTS = {"u1": ["i1", "i2", "i3"], "u2": ["i1", "i3", "i4"]}
RECOMMENDATIONS = frame(
    [(user, item, rank) for user, items in LISTS.items() for rank, item in enumerate(items, 1)],
    ["user", "item", "rank"],
)
USER_GROUPS = frame([("u1", "A"), ("u2", "B")], ["user", "group"])
ITEM_GROUPS = frame([("i1", "G1"), ("i2", "G1"), ("i3", "G2"), ("i4", "G2")], ["item", "group"])

# Exposure 1 / log2(1 + position) of positions 1, 2, 3.
PHI = (1.0, 1 / math.log2(3), 0.5)
TOTAL = 2 * sum(PHI)


class TestGroupDistribution:
    def test_logarithmic_exposure_of_item_groups(self):
        distribution = group_distribution(RECOMMENDATIONS, ITEM_GROUPS)
        assert isinstance(distribution, pd.Series)
        assert distribution.index.name == "group"
        assert distribution.name == "probability"
        assert list(distribution.index) == ["G1", "G2"]
        assert distribution["G1"] == pytest.approx((2 * PHI[0] + PHI[1]) / TOTAL)
        assert distribution["G1"] == pytest.approx(0.617320, abs=1e-6)
        assert distribution["G2"] == pytest.approx(0.382680, abs=1e-6)

    def test_flat_exposure_counts_recommendations(self):
        distribution = group_distribution(RECOMMENDATIONS, ITEM_GROUPS, exposure="flat")
        assert distribution.to_dict() == pytest.approx({"G1": 0.5, "G2": 0.5})

    def test_user_groups(self):
        distribution = group_distribution(RECOMMENDATIONS, USER_GROUPS)
        assert distribution.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})

    def test_cutoff(self):
        distribution = group_distribution(RECOMMENDATIONS, ITEM_GROUPS, k=1)
        assert distribution.to_dict() == pytest.approx({"G1": 1.0, "G2": 0.0})
        distribution = group_distribution(RECOMMENDATIONS, ITEM_GROUPS, k=2)
        assert distribution["G2"] == pytest.approx(PHI[1] / (2 * (PHI[0] + PHI[1])))

    def test_effectiveness_weights_exposure_by_relevance(self):
        # Relevant: (u1, i2) with grade 2 at position 2 and (u2, i4) with grade 1 at position 3.
        truth = frame(
            [("u1", "i2", 2.0), ("u2", "i4", 1.0), ("u2", "i9", 5.0)], ["user", "item", "rating"]
        )
        first, second = 2 * PHI[1], PHI[2]
        users = group_distribution(RECOMMENDATIONS, USER_GROUPS, ground_truth=truth)
        assert users.to_dict() == pytest.approx(
            {"A": first / (first + second), "B": second / (first + second)}
        )
        items = group_distribution(RECOMMENDATIONS, ITEM_GROUPS, ground_truth=truth)
        assert items.to_dict() == pytest.approx(
            {"G1": first / (first + second), "G2": second / (first + second)}
        )

    def test_binary_effectiveness_counts_hits_under_flat_exposure(self):
        truth = frame([("u1", "i1"), ("u1", "i3"), ("u2", "i1")], ["user", "item"])
        users = group_distribution(
            RECOMMENDATIONS, USER_GROUPS, exposure="flat", ground_truth=truth
        )
        assert users.to_dict() == pytest.approx({"A": 2 / 3, "B": 1 / 3})

    def test_groups_without_recommendations_have_zero_probability(self):
        groups = pd.concat([ITEM_GROUPS, frame([("i5", "G3")], ["item", "group"])])
        distribution = group_distribution(RECOMMENDATIONS, groups)
        assert distribution["G3"] == 0.0
        assert distribution.sum() == pytest.approx(1.0)

    def test_group_labels_can_be_integers(self):
        groups = ITEM_GROUPS.assign(group=[1, 1, 2, 2])
        assert list(group_distribution(RECOMMENDATIONS, groups).index) == [1, 2]


class TestEntropy:
    def test_logarithmic_exposure(self):
        p = (2 * PHI[0] + PHI[1]) / TOTAL
        expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        assert entropy(RECOMMENDATIONS, ITEM_GROUPS) == pytest.approx(expected)
        assert entropy(RECOMMENDATIONS, ITEM_GROUPS) == pytest.approx(0.959913, abs=1e-6)

    def test_uniform_distribution_has_maximum_entropy(self):
        assert entropy(RECOMMENDATIONS, ITEM_GROUPS, exposure="flat") == pytest.approx(1.0)
        assert entropy(RECOMMENDATIONS, USER_GROUPS) == pytest.approx(1.0)

    def test_single_group_has_no_entropy(self):
        assert entropy(RECOMMENDATIONS, ITEM_GROUPS, k=1) == 0.0

    def test_normalised_entropy_divides_by_log2_of_the_number_of_groups(self):
        groups = pd.concat([ITEM_GROUPS, frame([("i5", "G3"), ("i6", "G4")], ["item", "group"])])
        raw = entropy(RECOMMENDATIONS, groups)
        assert raw == pytest.approx(0.959913, abs=1e-6)
        assert entropy(RECOMMENDATIONS, groups, normalize=True) == pytest.approx(raw / 2)

    def test_normalised_entropy_of_a_single_group_is_undefined(self):
        groups = ITEM_GROUPS.assign(group="all")
        with pytest.raises(InvalidInputError, match="two groups"):
            entropy(RECOMMENDATIONS, groups, normalize=True)


class TestKullbackLeibler:
    def test_default_target_is_parity(self):
        # P(G1) = (2 phi_1 + phi_2) / total; against the parity target (1/2, 1/2),
        # KL = sum p log2(p / (1/2)).
        g1 = (2 * PHI[0] + PHI[1]) / TOTAL
        expected = g1 * math.log2(2 * g1) + (1 - g1) * math.log2(2 * (1 - g1))
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS) == pytest.approx(expected)
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS) == pytest.approx(0.040087, abs=1e-6)
        # With a uniform target over n groups, KL = log2(n) - H.
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS) == pytest.approx(
            1 - entropy(RECOMMENDATIONS, ITEM_GROUPS)
        )

    def test_no_divergence_from_itself(self):
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, exposure="flat") == pytest.approx(0.0)
        target = group_distribution(RECOMMENDATIONS, ITEM_GROUPS)
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, target) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "target",
        [
            {"G1": 0.8, "G2": 0.2},
            pd.Series({"G2": 0.2, "G1": 0.8}),
            frame([("G1", 0.8), ("G2", 0.2)], ["group", "probability"]),
        ],
        ids=["mapping", "series", "frame"],
    )
    def test_direction_is_model_from_target(self, target):
        p = (2 * PHI[0] + PHI[1]) / TOTAL
        expected = p * math.log2(p / 0.8) + (1 - p) * math.log2((1 - p) / 0.2)
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, target) == pytest.approx(expected)

    def test_target_group_without_recommendations(self):
        # P(G3) = 0 contributes nothing: KL = sum_x P(x) log2(P(x) / Q(x)) over P(x) > 0.
        p = (2 * PHI[0] + PHI[1]) / TOTAL
        target = {"G1": 0.5, "G2": 0.25, "G3": 0.25}
        expected = p * math.log2(p / 0.5) + (1 - p) * math.log2((1 - p) / 0.25)
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, target) == pytest.approx(expected)

    def test_infinite_when_the_target_excludes_a_recommended_group(self):
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, {"G1": 1.0, "G2": 0.0}) == math.inf

    def test_invalid_targets(self):
        with pytest.raises(InvalidInputError, match="sum to 1"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, {"G1": 0.5, "G2": 0.6})
        with pytest.raises(InvalidInputError, match="G2"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, {"G1": 1.0})
        with pytest.raises(InvalidInputError, match="non-negative"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, {"G1": 1.5, "G2": -0.5})
        with pytest.raises(MissingColumnsError, match="probability"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, frame([("G1", 1.0)], ["group", "share"]))


class TestMutualInformation:
    def test_user_groups_and_item_groups(self):
        value = mutual_information(
            RECOMMENDATIONS, user_groups=USER_GROUPS, item_groups=ITEM_GROUPS
        )
        joint = {
            ("A", "G1"): (PHI[0] + PHI[1]) / TOTAL,
            ("A", "G2"): PHI[2] / TOTAL,
            ("B", "G1"): PHI[0] / TOTAL,
            ("B", "G2"): (PHI[1] + PHI[2]) / TOTAL,
        }
        p_g1 = joint["A", "G1"] + joint["B", "G1"]
        expected = sum(
            p * math.log2(p / (0.5 * (p_g1 if item == "G1" else 1 - p_g1)))
            for (_, item), p in joint.items()
        )
        assert value == pytest.approx(expected)
        assert value == pytest.approx(0.068270, abs=1e-6)

    def test_flat_exposure(self):
        value = mutual_information(
            RECOMMENDATIONS, user_groups=USER_GROUPS, item_groups=ITEM_GROUPS, exposure="flat"
        )
        # Six recommendations: A has 2 in G1 and 1 in G2, B the opposite; every marginal is 1/2.
        expected = 2 * (2 / 6) * math.log2((2 / 6) / 0.25) + 2 * (1 / 6) * math.log2((1 / 6) / 0.25)
        assert value == pytest.approx(expected)
        assert value == pytest.approx(0.081704, abs=1e-6)

    def test_individual_users_against_item_groups(self):
        # Every user is their own block; here that coincides with the user groups.
        value = mutual_information(RECOMMENDATIONS, item_groups=ITEM_GROUPS)
        assert value == pytest.approx(0.068270, abs=1e-6)

    def test_user_groups_against_individual_items(self):
        value = mutual_information(RECOMMENDATIONS, user_groups=USER_GROUPS, exposure="flat")
        # Joint over (group, item): A sees i1, i2, i3 and B sees i1, i3, i4, 1/6 each.
        # i1 and i3 are shared (independent of the group), i2 and i4 identify it.
        expected = 4 * (1 / 6) * math.log2((1 / 6) / (0.5 * (1 / 3))) + 2 * (1 / 6) * math.log2(
            (1 / 6) / (0.5 * (1 / 6))
        )
        assert value == pytest.approx(expected)
        assert value == pytest.approx(1 / 3)

    def test_identical_lists_are_independent_of_the_user(self):
        recommendations = frame(
            [("u1", "i1", 1), ("u1", "i3", 2), ("u2", "i1", 1), ("u2", "i3", 2)],
            ["user", "item", "rank"],
        )
        value = mutual_information(
            recommendations, user_groups=USER_GROUPS, item_groups=ITEM_GROUPS
        )
        assert value == pytest.approx(0.0, abs=1e-12)

    def test_segregated_lists_are_fully_dependent(self):
        recommendations = frame(
            [("u1", "i1", 1), ("u1", "i2", 2), ("u2", "i3", 1), ("u2", "i4", 2)],
            ["user", "item", "rank"],
        )
        value = mutual_information(
            recommendations, user_groups=USER_GROUPS, item_groups=ITEM_GROUPS
        )
        assert value == pytest.approx(1.0)

    def test_effectiveness(self):
        truth = frame([("u1", "i1"), ("u2", "i4")], ["user", "item"])
        value = mutual_information(
            RECOMMENDATIONS, user_groups=USER_GROUPS, item_groups=ITEM_GROUPS, ground_truth=truth
        )
        # All the effectiveness of A is on G1 and all that of B on G2: I = H(user group).
        p = PHI[0] / (PHI[0] + PHI[2])
        assert value == pytest.approx(-p * math.log2(p) - (1 - p) * math.log2(1 - p))

    def test_needs_at_least_one_grouping(self):
        with pytest.raises(InvalidInputError, match="user_groups"):
            mutual_information(RECOMMENDATIONS)


class TestErrors:
    def test_groups_must_identify_one_side(self):
        with pytest.raises(MissingColumnsError, match="group"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS[["item"]])
        with pytest.raises(InvalidInputError, match="either"):
            entropy(RECOMMENDATIONS, frame([("x", "g")], ["movie", "group"]))
        with pytest.raises(InvalidInputError, match="either"):
            entropy(RECOMMENDATIONS, frame([("u1", "i1", "g")], ["user", "item", "group"]))

    def test_wrong_side(self):
        with pytest.raises(MissingColumnsError, match="user"):
            mutual_information(RECOMMENDATIONS, user_groups=ITEM_GROUPS)

    def test_every_recommended_entity_needs_a_group(self):
        with pytest.raises(InvalidInputError, match="1 recommended item"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS.iloc[:3])

    def test_an_entity_cannot_be_in_two_groups(self):
        groups = pd.concat([ITEM_GROUPS, frame([("i1", "G2")], ["item", "group"])])
        with pytest.raises(InvalidInputError, match="more than once"):
            entropy(RECOMMENDATIONS, groups)

    def test_missing_group(self):
        with pytest.raises(InvalidInputError, match="missing"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS.assign(group=["G1", None, "G2", "G2"]))

    def test_invalid_options(self):
        with pytest.raises(InvalidInputError, match="exposure"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS, exposure="linear")
        with pytest.raises(InvalidInputError, match="k must be a positive integer"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS, k=0)

    def test_no_benefit_to_distribute(self):
        truth = frame([("u1", "i4")], ["user", "item"])
        with pytest.raises(InvalidInputError, match="the total benefit is zero"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS, ground_truth=truth)
        with pytest.raises(InvalidInputError, match="the total benefit is zero"):
            entropy(RECOMMENDATIONS.iloc[:0], ITEM_GROUPS)


@pytest.mark.parametrize("k", [1, 5, None])
@pytest.mark.parametrize("exposure", ["logarithmic", "flat"])
@pytest.mark.parametrize("effectiveness", [False, True])
def test_agrees_with_the_reference_implementation(random_data, k, exposure, effectiveness):
    recommendations = random_data["recommendations"]
    truth = random_data["ground_truth"] if effectiveness else None
    user_groups, item_groups = random_data["user_groups"], random_data["item_groups"]

    lists = lists_from_rows(recommendations.itertuples(index=False))
    relevance = None
    if effectiveness:
        relevance = {}
        for user, item, grade in truth.itertuples(index=False):
            relevance.setdefault(user, {})[item] = grade
    user_group = dict(user_groups.itertuples(index=False, name=None))
    item_group = dict(item_groups.itertuples(index=False, name=None))
    joint = joint_distribution(
        lists, k, user_group.get, item_group.get, flat=exposure == "flat", relevance=relevance
    )
    options = {"k": k, "exposure": exposure, "ground_truth": truth}

    items = marginal(joint, 1)
    actual = group_distribution(recommendations, item_groups, **options)
    assert actual.to_dict() == pytest.approx(
        {group: items.get(group, 0.0) for group in actual.index}
    )
    assert entropy(recommendations, item_groups, **options) == pytest.approx(
        reference_entropy(items)
    )
    assert entropy(recommendations, user_groups, **options) == pytest.approx(
        reference_entropy(marginal(joint, 0))
    )

    target = {"head": 0.6, "mid": 0.3, "tail": 0.1}
    assert kl_divergence(recommendations, item_groups, target, **options) == pytest.approx(
        reference_kl(items, target)
    )

    assert mutual_information(
        recommendations, user_groups=user_groups, item_groups=item_groups, **options
    ) == pytest.approx(reference_mi(joint))
    by_user = joint_distribution(
        lists, k, lambda user: user, item_group.get, flat=exposure == "flat", relevance=relevance
    )
    assert mutual_information(recommendations, item_groups=item_groups, **options) == pytest.approx(
        reference_mi(by_user)
    )
    by_item = joint_distribution(
        lists, k, user_group.get, lambda item: item, flat=exposure == "flat", relevance=relevance
    )
    assert mutual_information(recommendations, user_groups=user_groups, **options) == pytest.approx(
        reference_mi(by_item)
    )


class TestReviewRegressions:
    """Inputs that used to give a wrong value or an undocumented exception."""

    def test_target_probabilities_must_be_numeric(self):
        with pytest.raises(InvalidInputError, match=r"target.*numeric"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, {"G1": "x", "G2": 1.0})

    def test_normalised_entropy_never_exceeds_one(self):
        # Eleven groups with the same exposure: H = log2(11), so the ratio is exactly 1.
        recommendations = frame(
            [("u", item, item) for item in range(1, 12)], ["user", "item", "rank"]
        )
        groups = frame([(item, item) for item in range(1, 12)], ["item", "group"])
        assert entropy(recommendations, groups, exposure="flat", normalize=True) == 1.0

    def test_dependence_on_many_users_and_many_groups(self):
        # One recommendation per user, one group per item: I = H = log2(n). The joint has
        # n cells of benefit; it must not need n * n of memory.
        n = 50_000
        ids = list(range(n))
        recommendations = pd.DataFrame({"user": ids, "item": ids})
        groups = pd.DataFrame({"item": ids, "group": ids})
        value = mutual_information(recommendations, item_groups=groups, exposure="flat")
        assert value == pytest.approx(math.log2(n))


class TestEdgeCases:
    def test_target_must_list_every_group_once(self):
        target = pd.Series([0.5, 0.5], index=["G1", "G1"])
        with pytest.raises(InvalidInputError, match="more than once"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, target)

    def test_group_labels_of_different_types(self):
        groups = frame([("i1", "G1"), ("i2", "G1"), ("i3", 2), ("i4", 2)], ["item", "group"])
        distribution = group_distribution(RECOMMENDATIONS, groups, exposure="flat")
        assert distribution.to_dict() == pytest.approx({"G1": 0.5, 2: 0.5})

    def test_cutoff_beyond_the_lists_adds_nothing(self):
        for metric in (group_distribution, entropy, kl_divergence):
            full = metric(RECOMMENDATIONS, ITEM_GROUPS)
            cut = metric(RECOMMENDATIONS, ITEM_GROUPS, k=100)
            assert (cut.to_dict() if isinstance(cut, pd.Series) else cut) == pytest.approx(
                full.to_dict() if isinstance(full, pd.Series) else full
            )
        assert mutual_information(RECOMMENDATIONS, item_groups=ITEM_GROUPS, k=100) == pytest.approx(
            mutual_information(RECOMMENDATIONS, item_groups=ITEM_GROUPS)
        )

    def test_group_labels_must_be_hashable(self):
        groups = frame([("i1", [1]), ("i2", [1]), ("i3", [2]), ("i4", [2])], ["item", "group"])
        with pytest.raises(InvalidInputError, match="hashable"):
            entropy(RECOMMENDATIONS, groups)

    def test_group_table_with_a_missing_identifier(self):
        groups = pd.concat([ITEM_GROUPS, frame([(None, "G1")], ["item", "group"])])
        with pytest.raises(InvalidInputError, match="identifiers must not be missing"):
            entropy(RECOMMENDATIONS, groups)

    def test_ground_truth_is_validated_before_it_is_restricted(self):
        # The row without a user concerns no recommended user, but is still invalid.
        truth = frame([("u1", "i2"), (None, "i3")], ["user", "item"])
        with pytest.raises(InvalidInputError, match="user identifiers must not be missing"):
            entropy(RECOMMENDATIONS, ITEM_GROUPS, ground_truth=truth)

    def test_ground_truth_of_another_identifier_type(self):
        recommendations = frame([(1, 10, 1), (2, 11, 1)], ["user", "item", "rank"])
        groups = frame([(10, "a"), (11, "b")], ["item", "group"])
        truth = frame([(1, 10), (2, 11)], ["user", "item"])
        assert entropy(recommendations, groups, ground_truth=truth) == pytest.approx(1.0)
        with pytest.raises(InvalidInputError, match=r"integer.*string|string.*integer"):
            entropy(recommendations, groups, ground_truth=truth.astype({"user": str}))
        with pytest.raises(InvalidInputError, match=r"integer.*string|string.*integer"):
            entropy(recommendations, groups, ground_truth=truth.astype({"item": str}))

    def test_identifiers_must_be_hashable_with_a_ground_truth(self):
        recommendations = frame([("u1", ["x"]), ("u2", ["y"])], ["user", "item"])
        truth = frame([("u1", "p")], ["user", "item"])
        with pytest.raises(InvalidInputError, match="item identifiers must be hashable"):
            kl_divergence(recommendations, USER_GROUPS, ground_truth=truth)

    @pytest.mark.parametrize(
        ("rows", "message"),
        [
            ([("u1", "i2", 1.0), ("u9", "i9", -1.0)], "relevance must be finite and non-negative"),
            ([("u1", "i2", 1.0), ("u9", "i9", float("nan"))], "relevance must be finite"),
            ([("u1", "i2", 1.0), ("u9", "i9", 1.0), ("u9", "i9", 1.0)], "more than once"),
        ],
    )
    def test_the_whole_ground_truth_is_validated(self, rows, message):
        # The invalid rows concern no recommended user or item: the table is still invalid.
        truth = frame(rows, ["user", "item", "rating"])
        with pytest.raises(InvalidInputError, match=message):
            entropy(RECOMMENDATIONS, ITEM_GROUPS, ground_truth=truth)

    def test_ground_truth_about_other_users_and_items_changes_nothing(self):
        truth = frame([("u1", "i2", 2.0), ("u2", "i4", 1.0)], ["user", "item", "rating"])
        more = pd.concat(
            [truth, frame([("u9", "i1", 5.0), ("u1", "i9", 5.0)], ["user", "item", "rating"])]
        )
        for groups in (ITEM_GROUPS, USER_GROUPS):
            pd.testing.assert_series_equal(
                group_distribution(RECOMMENDATIONS, groups, ground_truth=more),
                group_distribution(RECOMMENDATIONS, groups, ground_truth=truth),
            )
        assert mutual_information(
            RECOMMENDATIONS, item_groups=ITEM_GROUPS, ground_truth=more
        ) == mutual_information(RECOMMENDATIONS, item_groups=ITEM_GROUPS, ground_truth=truth)

    def test_target_order_does_not_change_the_result(self):
        # Five groups of four items; the same target with its rows in every rotation.
        recommendations = frame(
            [(user, (3 * user + 7 * rank) % 20, rank) for user in range(9) for rank in range(1, 6)],
            ["user", "item", "rank"],
        )
        groups = frame([(item, item % 5) for item in range(20)], ["item", "group"])
        target = frame(list(enumerate([0.1, 0.2, 0.3, 0.15, 0.25])), ["group", "probability"])
        values = {
            kl_divergence(recommendations, groups, target.sample(frac=1, random_state=seed))
            for seed in range(10)
        }
        assert len(values) == 1

    def test_group_table_must_describe_one_side(self):
        both = frame([("u1", "i1", "A"), ("u2", "i2", "B")], ["user", "item", "group"])
        with pytest.raises(InvalidInputError, match="either a 'user' or an 'item' column"):
            mutual_information(RECOMMENDATIONS, user_groups=both)
        with pytest.raises(InvalidInputError, match="either a 'user' or an 'item' column"):
            mutual_information(RECOMMENDATIONS, item_groups=both)

    def test_target_sums_to_one_within_a_millionth(self):
        # Documented tolerance on the sum of the target: 1e-6.
        close = {"G1": 0.5, "G2": 0.5 + 5e-7}
        assert kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, close) == pytest.approx(
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS), abs=1e-5
        )
        with pytest.raises(InvalidInputError, match="must sum to 1"):
            kl_divergence(RECOMMENDATIONS, ITEM_GROUPS, {"G1": 0.5, "G2": 0.5 + 5e-6})

    def test_nothing_recommended_and_no_group(self):
        # Only the ground truth has users and items: there is still nothing to distribute.
        truth = frame([("u1", "i1")], ["user", "item"])
        with pytest.raises(InvalidInputError, match="the total benefit is zero"):
            entropy(RECOMMENDATIONS.iloc[:0], ITEM_GROUPS.iloc[:0], ground_truth=truth)


class TestScores:
    def test_scores_order_the_lists(self):
        scored = RECOMMENDATIONS.assign(score=-RECOMMENDATIONS["rank"]).drop(columns="rank")
        pd.testing.assert_series_equal(
            group_distribution(scored, ITEM_GROUPS),
            group_distribution(RECOMMENDATIONS, ITEM_GROUPS),
        )

    def test_tied_scores_share_the_exposure(self):
        # i1 (group G1) and i3 (group G2) have the same score: with a cutoff of one,
        # each is expected to receive half of the exposure of the first position.
        scored = frame(
            [("u1", "i1", 5.0), ("u1", "i3", 5.0), ("u1", "i2", 1.0)], ["user", "item", "score"]
        )
        distribution = group_distribution(scored, ITEM_GROUPS, k=1)
        assert distribution.to_dict() == pytest.approx({"G1": 0.5, "G2": 0.5})
        # Whole list: the tied items expect the mean exposure of positions 1 and 2.
        shared = (PHI[0] + PHI[1]) / 2
        distribution = group_distribution(scored, ITEM_GROUPS)
        assert distribution["G2"] == pytest.approx(shared / (2 * shared + PHI[2]))

    @pytest.mark.parametrize("effectiveness", [False, True])
    @pytest.mark.parametrize("exposure", ["logarithmic", "flat"])
    @pytest.mark.parametrize("k", [2, 9, None])
    def test_agrees_with_the_average_over_every_order_of_the_ties(
        self, tied_data, k, exposure, effectiveness
    ):
        from conftest import scored_lists
        from reference import expected_joint_distribution

        recommendations = tied_data["recommendations"]
        truth = tied_data["ground_truth"] if effectiveness else None
        relevance = None
        if effectiveness:
            relevance = {}
            for user, item, grade in truth.itertuples(index=False):
                relevance.setdefault(user, {})[item] = grade
        user_group = dict(tied_data["user_groups"].itertuples(index=False, name=None))
        item_group = dict(tied_data["item_groups"].itertuples(index=False, name=None))
        lists = scored_lists(recommendations)
        options = {"k": k, "exposure": exposure, "ground_truth": truth}

        def joint(block_of_user, block_of_item):
            return expected_joint_distribution(
                lists, k, block_of_user, block_of_item, flat=exposure == "flat", relevance=relevance
            )

        grouped = joint(user_group.get, item_group.get)
        actual = group_distribution(recommendations, tied_data["item_groups"], **options)
        items = marginal(grouped, 1)
        assert actual.to_dict() == pytest.approx({g: items.get(g, 0.0) for g in actual.index})
        assert mutual_information(
            recommendations,
            user_groups=tied_data["user_groups"],
            item_groups=tied_data["item_groups"],
            **options,
        ) == pytest.approx(reference_mi(grouped), abs=1e-12)
        assert mutual_information(
            recommendations, item_groups=tied_data["item_groups"], **options
        ) == pytest.approx(reference_mi(joint(lambda user: user, item_group.get)), abs=1e-12)
        assert mutual_information(
            recommendations, user_groups=tied_data["user_groups"], **options
        ) == pytest.approx(reference_mi(joint(user_group.get, lambda item: item)), abs=1e-12)
