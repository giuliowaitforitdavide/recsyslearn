"""What recsyslearn refuses, and how to handle it.

A table that does not mean what a metric needs is refused with an error that
says what is wrong and in which argument, instead of producing a number.
"""

import pandas as pd

from recsyslearn import fairness
from recsyslearn.accuracy import ndcg
from recsyslearn.beyond_accuracy import coverage
from recsyslearn.errors import (
    InvalidInputError,
    MissingColumnsError,
    RecsyslearnError,
)

recommendations = pd.DataFrame(
    {
        "user": ["ann", "ann", "bob"],
        "item": ["x", "y", "x"],
        "rank": [1, 2, 1],
    }
)
truth = pd.DataFrame({"user": ["ann", "bob"], "item": ["y", "x"]})
groups = pd.DataFrame({"item": ["x", "y"], "group": ["head", "tail"]})


def refused(description: str, call) -> None:
    """Run ``call``, which is expected to be refused, and print why."""
    try:
        call()
    except RecsyslearnError as error:
        print(f"{description}\n  {type(error).__name__}: {error}")
    else:
        raise AssertionError(f"accepted: {description}")


# A column is missing: the error carries the name of the argument and the
# columns.
try:
    ndcg(recommendations.rename(columns={"item": "movie"}), truth)
except MissingColumnsError as error:
    print(f"missing columns of {error.name}: {error.missing}")

# The content is wrong: InvalidInputError.
refused(
    "two items of a user with the same rank",
    lambda: ndcg(recommendations.assign(rank=[1, 1, 1]), truth),
)
refused(
    "the same item recommended twice to a user",
    lambda: ndcg(recommendations.assign(item=["x", "x", "x"]), truth),
)
refused(
    "a missing score",
    lambda: ndcg(
        recommendations.drop(columns="rank").assign(score=[0.5, None, 0.1]),
        truth,
    ),
)
refused(
    "ranks that are not numbers",
    lambda: ndcg(recommendations.assign(rank=["1", "2", "1"]), truth),
)
refused(
    "a missing identifier",
    lambda: ndcg(recommendations.assign(user=["ann", None, "bob"]), truth),
)
refused(
    "identifiers of different types in two tables (integers and strings)",
    lambda: ndcg(recommendations.assign(item=[1, 2, 1]), truth),
)
refused(
    "a cutoff that is not a positive integer",
    lambda: ndcg(recommendations, truth, k=0),
)
refused(
    "a recommended item that is not in the catalog",
    lambda: coverage(recommendations, ["x"]),
)
refused(
    "a recommended item without a group",
    lambda: fairness.entropy(recommendations, groups.head(1)),
)
refused(
    "a target that does not sum to 1",
    lambda: fairness.kl_divergence(
        recommendations, groups, {"head": 0.5, "tail": 0.6}
    ),
)
refused(
    "tied scores across the cutoff, for coverage",
    lambda: coverage(
        recommendations.drop(columns="rank").assign(score=[1.0, 1.0, 1.0]),
        ["x", "y"],
        k=1,
    ),
)

# Both errors are also ValueError, and both derive from RecsyslearnError.
print(f"\nValueError: {issubclass(InvalidInputError, ValueError)}", end=", ")
print(f"{issubclass(MissingColumnsError, ValueError)}")
