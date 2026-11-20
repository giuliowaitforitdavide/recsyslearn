"""Reproduce Tables 5 and 6 of

    E. Amigó, Y. Deldjoo, S. Mizzaro, A. Bellogín. "A unifying and general account of
    fairness measurement in recommender systems". Information Processing & Management
    60(1), 2023. https://doi.org/10.1016/j.ipm.2022.103115

from the files the authors publish with their reference implementation
(https://github.com/abellogin/FairnessFramework4RecSys, ``results/soa_recsys_experiment``):
the top-10 recommendations of ten systems on CiteULike and on a Netflix sample, the test
ratings, and the popularity class of every item and activity class of every user.

Two references are checked: the tables printed in the paper (eight systems, fifteen
fairness columns) and the output of the authors' script stored in the repository (ten
systems, eighteen columns).

How the columns map onto recsyslearn (as read in the authors' ``eval.pl``):

* exposure is ``1 / log2(1 + rank)``; effectiveness multiplies it by the test rating;
* "Un" is parity (the default target), "Pr" is proportionality to the size of each
  group, "UtilEq" is proportionality to the utility of each group, i.e. to its share of
  the test ratings;
* the three mutual informations are between user groups and item groups, individual
  users and item groups, user groups and individual items.

The authors' script truncates, rather than rounds, to three decimals (four for the
mutual information between groups, which the paper then rounds to three).
"""

from __future__ import annotations

import functools
import itertools
import math

import pandas as pd
import pytest
from paper_data import fetch

from recsyslearn import fairness

pytestmark = pytest.mark.integration

REPOSITORY = "abellogin/FairnessFramework4RecSys"
COMMIT = "91f4e15987480b22884e63cc5a9a9bcb2fc1ac2b"
FOLDER = "results/soa_recsys_experiment"
DATASETS = {
    "citeulike": ("DATASET1", "citeulike-a", "res.citeulike.txt"),
    "netflix": ("DATASET2", "netflix-small", "res.netflix.txt"),
}
SYSTEMS = ["BPR", "MF", "MostPop", "NeuMF", "PMF", "Popular", "Random", "UnPopular", "VAECF", "WMF"]

# The columns of Tables 5 and 6 after the DCG one, under the names of the authors' script.
# Formatting is switched off for the tables below: one line per group of columns or row.
COLUMNS = [
    "KL-U-Eff-Un", "KL-U-Eff-Pr", "KL-U-Ef-UtilEq",
    "KL-I-Exp-Un", "KL-I-Eff-Un", "KL-I-Exp-Pr", "KL-I-Eff-Pr", "KL-I-Exp-UtilEq", "KL-I-Ef-UtilEq",
    "MI-U-I-Exp", "MI-U-I-Eff", "MI-I-u-Exp", "MI-I-u-Eff", "MI-i-U-Exp", "MI-i-U-Eff",
]  # fmt: skip
# Truncated to four decimals by the script, then rounded to three in the paper.
FOUR_DECIMALS = {"MI-U-I-Exp", "MI-U-I-Eff"}
# Row of the paper -> recommendation file. The "Oracle" row is left out: the script
# builds it following the iteration order of a Perl hash, which is not reproducible.
ROWS = {
    "Random": "Random", "Popularity": "MostPop", "MF": "MF", "PMF": "PMF",
    "BPR-MF": "BPR", "WMF": "WMF", "NeuMF": "NeuMF", "VAECF": "VAECF",
}  # fmt: skip
PRINTED = {
    # Table 5
    "citeulike": {
        "Random": "0.210 0.219 0.005 0.023 0.357 0.000 0.204 0.171 0.005 0.000 0.131 0.091 0.642 0.305 0.789",
        "Popularity": "0.227 0.237 0.003 1.000 1.000 0.760 0.760 0.316 0.316 0.000 0.000 0.000 0.000 0.000 0.032",
        "MF": "0.118 0.125 0.040 0.001 0.409 0.036 0.244 0.376 0.014 0.000 0.000 0.091 0.590 0.042 0.881",
        "PMF": "0.064 0.069 0.089 0.975 0.977 0.736 0.739 0.296 0.298 0.000 0.003 0.019 0.022 0.031 0.423",
        "BPR-MF": "0.227 0.237 0.003 1.000 1.000 0.760 0.760 0.316 0.316 0.000 0.000 0.000 0.000 0.000 0.032",
        "WMF": "0.186 0.195 0.011 0.501 0.707 0.319 0.494 0.039 0.127 0.009 0.014 0.152 0.237 0.246 0.646",
        "NeuMF": "0.108 0.115 0.047 0.786 0.884 0.564 0.653 0.171 0.232 0.002 0.002 0.125 0.108 0.115 0.370",
        "VAECF": "0.102 0.109 0.051 0.864 0.916 0.634 0.682 0.219 0.254 0.000 0.006 0.087 0.078 0.067 0.340",
    },
    # Table 6
    "netflix": {
        "Random": "0.307 0.666 0.001 0.450 0.313 0.000 1.782 1.593 0.000 0.000 0.019 0.094 0.642 0.051 0.584",
        "Popularity": "0.026 0.185 0.145 1.000 1.000 2.977 2.977 0.300 0.300 0.000 0.000 0.000 0.000 0.000 0.017",
        "MF": "0.344 0.718 0.006 0.108 0.757 0.162 2.623 0.867 0.142 0.020 0.000 0.161 0.221 0.067 0.066",
        "PMF": "0.172 0.464 0.011 0.012 0.564 0.414 2.291 0.508 0.054 0.011 0.001 0.222 0.402 0.046 0.135",
        "BPR-MF": "0.043 0.226 0.110 1.000 1.000 2.977 2.977 0.300 0.300 0.000 0.000 0.000 0.000 0.000 0.020",
        "WMF": "0.034 0.206 0.126 0.497 0.818 2.167 2.718 0.032 0.176 0.000 0.002 0.214 0.161 0.150 0.274",
        "NeuMF": "0.000 0.085 0.275 0.868 0.933 2.794 2.888 0.207 0.250 0.000 0.000 0.094 0.062 0.092 0.156",
        "VAECF": "0.012 0.143 0.189 0.914 0.971 2.861 2.941 0.237 0.278 0.000 0.000 0.071 0.027 0.115 0.171",
    },
}  # fmt: skip


def read(dataset: str, name: str, columns: dict[str, str]) -> pd.DataFrame:
    folder, tag, _ = DATASETS[dataset]
    path = fetch(REPOSITORY, COMMIT, f"{FOLDER}/{folder}/{name.format(tag=tag)}")
    return pd.read_csv(path, usecols=list(columns)).rename(columns=columns)[list(columns.values())]


@functools.cache
def inputs(dataset: str):
    """Test ratings, group tables and the two proportional targets of each side."""
    item_groups = read(
        dataset, "{tag}_item_w_popClass.csv", {"itemId": "item", "popClass": "group"}
    )
    user_groups = read(
        dataset, "{tag}_users_w_activClass.csv", {"userId": "user", "is_active": "group"}
    )
    truth = read(
        dataset,
        "{tag}_ratings_df_test.csv",
        {"userId": "user", "itemId": "item", "rating": "rating"},
    )
    # As the authors' script: a repeated pair keeps its last rating, and only the items
    # that have a popularity class are considered.
    truth = truth.drop_duplicates(["user", "item"], keep="last")
    truth = truth[truth["item"].isin(item_groups["item"])]

    targets = {}
    for side, groups in (("U", user_groups), ("I", item_groups)):
        targets[side, "Pr"] = groups["group"].value_counts(normalize=True)
        # Share of the test ratings of each group: the distribution of the utility.
        targets[side, "UtilEq"] = fairness.group_distribution(
            truth[["user", "item"]], groups, exposure="flat", ground_truth=truth
        )
    return truth, user_groups, item_groups, targets


@functools.cache
def computed(dataset: str, system: str, scored: bool = False) -> dict[str, float]:
    """Every column of the authors' output for one system, computed with recsyslearn.

    With ``scored``, the published ranks are passed as scores (the negated rank) instead
    of as ranks: the lists are the same, and so must the results be.
    """
    truth, user_groups, item_groups, targets = inputs(dataset)
    recommendations = read(
        dataset,
        system + "_{tag}_samplingRate_1.csv",
        {"userId": "user", "itemId": "item", "rank": "rank"},
    ).drop_duplicates(["user", "item"], keep="last")
    if scored:
        recommendations = recommendations.assign(score=-recommendations["rank"]).drop(
            columns="rank"
        )

    values = {}
    for benefit, ground_truth in (("Exp", None), ("Eff", truth)):
        options = {"exposure": "logarithmic", "ground_truth": ground_truth}
        for side, groups in (("U", user_groups), ("I", item_groups)):
            kl = functools.partial(fairness.kl_divergence, recommendations, groups, **options)
            values[f"KL-{side}-{benefit}-Un"] = kl()
            values[f"KL-{side}-{benefit}-Pr"] = kl(targets[side, "Pr"])
            # The script spells this column "Exp-UtilEq" but "Ef-UtilEq".
            name = f"KL-{side}-{'Ef' if benefit == 'Eff' else benefit}-UtilEq"
            values[name] = kl(targets[side, "UtilEq"])
        mi = functools.partial(fairness.mutual_information, recommendations, **options)
        values[f"MI-U-I-{benefit}"] = mi(user_groups=user_groups, item_groups=item_groups)
        values[f"MI-I-u-{benefit}"] = mi(item_groups=item_groups)
        values[f"MI-i-U-{benefit}"] = mi(user_groups=user_groups)
    return values


def truncate(value: float, decimals: int) -> float:
    """What the authors' script prints: ``int(x * 10^d) / 10^d``.

    A value within 1e-9 of the next multiple is taken as that multiple, so that the
    comparison does not hinge on the last bit of a floating point sum.
    """
    scaled = value * 10**decimals
    if round(scaled) - scaled < 1e-9 * 10**decimals:
        scaled = round(scaled)
    return math.floor(scaled) / 10**decimals


@functools.cache
def stored(dataset: str) -> dict[str, dict[str, float]]:
    """The rows of the output of the authors' script kept in their repository."""
    lines = fetch(REPOSITORY, COMMIT, f"{FOLDER}/{DATASETS[dataset][2]}").read_text().splitlines()
    rows = {}
    for header, row in itertools.pairwise(lines):
        if header.startswith("SYSTEM\tEFF"):
            names, cells = header.split("\t")[1:], row.split("\t")
            rows[cells[0].split("_")[0]] = dict(zip(names, map(float, cells[1:]), strict=True))
    return rows


@pytest.mark.parametrize("row", list(ROWS))
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_values_printed_in_the_paper(dataset, row):
    values = computed(dataset, ROWS[row])
    printed = dict(zip(COLUMNS, map(float, PRINTED[dataset][row].split()), strict=True))
    shown = {
        column: round(truncate(values[column], 4), 3)
        if column in FOUR_DECIMALS
        else truncate(values[column], 3)
        for column in COLUMNS
    }
    assert shown == printed


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_output_of_the_reference_implementation(dataset, system):
    values = computed(dataset, system)
    reference = stored(dataset)[system]
    assert set(values) <= set(reference)
    shown = {
        column: truncate(value, 4 if column in FOUR_DECIMALS else 3)
        for column, value in values.items()
    }
    assert shown == {column: reference[column] for column in values}


@pytest.mark.parametrize("system", ["BPR", "Random"])
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_scores_reproduce_the_same_values_as_ranks(dataset, system):
    """The lists given as scores instead of ranks: every column is identical, hence
    equal to the published one."""
    assert computed(dataset, system, scored=True) == computed(dataset, system)
