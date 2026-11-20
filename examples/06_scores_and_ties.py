"""Recommendations with scores, and what happens when scores are tied.

A popularity baseline is the typical case: it scores an item with the number of users
who interacted with it, so many items have exactly the same score.
"""

import pandas as pd

from recsyslearn import fairness, segmentation
from recsyslearn.accuracy import ndcg
from recsyslearn.beyond_accuracy import coverage, novelty
from recsyslearn.errors import InvalidInputError

train = pd.read_csv("./examples/__assets__/train_dataset.csv")
test = pd.read_csv("./examples/__assets__/test_dataset.csv")

# The candidates of every user: the 150 most popular items, minus those the user already
# interacted with. The score is the popularity.
popularity = train["item"].value_counts().rename("score").rename_axis("item").reset_index()
candidates = pd.DataFrame({"user": train["user"].unique()}).merge(
    popularity.nlargest(150, "score", keep="all"), how="cross"
)
seen = candidates.merge(train, on=["user", "item"], how="left", indicator=True)["_merge"]
scored = candidates[(seen == "left_only").to_numpy()]

position = scored.groupby("user")["score"].rank(method="first", ascending=False)
tied = scored.groupby("user")["score"].transform(lambda scores: scores.duplicated(keep=False))
share = tied[position <= 10].mean()
print(f"candidates in the first 10 positions that are tied with another: {share:.1%}")

# 1. A table with a `score` column is read as it is. The order of tied items is unknown,
#    and recsyslearn does not choose one: nDCG is its mean over all the orders of the
#    ties, computed exactly (McSherry & Najork, 2008).
print(f"\nexpected nDCG@10 over the orders of the ties: {ndcg(scored, test, k=10):.4f}")

# 2. To impose an order instead, compute the ranks and pass them. Two ways of breaking
#    the ties give two different values, and the expected one lies between them.
ordered = scored.sort_values(["user", "score", "item"], ascending=[True, False, True])
low_ids_first = ordered.assign(rank=ordered.groupby("user").cumcount() + 1)
ordered = scored.sort_values(["user", "score", "item"], ascending=[True, False, False])
high_ids_first = ordered.assign(rank=ordered.groupby("user").cumcount() + 1)
print(f"ties broken by ascending identifier:          {ndcg(low_ids_first, test, k=10):.4f}")
print(f"ties broken by descending identifier:         {ndcg(high_ids_first, test, k=10):.4f}")

# 3. Novelty with ties: an item tied across the cutoff counts for the probability that
#    it falls within it.
print(f"\nexpected novelty@10: {novelty(scored, train, k=10):.4f} bits")

# 4. Fairness with ties: every tied item gets the exposure it is expected to receive,
#    and the metrics are computed on those expected exposures.
item_groups = segmentation.by_cumulative_interactions(train, (0.2, 0.8), labels=("head", "rest"))
distribution = fairness.group_distribution(scored, item_groups, k=10)
print(f"expected exposure of the head items in the top 10: {distribution['head']:.4f}")

# 5. Coverage is the exception: which items are recommended is not defined when a tie
#    straddles the cutoff, and no published definition says how to count them. The
#    table is refused instead of being given an arbitrary order.
try:
    coverage(scored, train, k=10)
except InvalidInputError as error:
    print(f"\ncoverage@10 from the scores is refused:\n  {error}")

# With an order, or with a cutoff that splits no tie (here: none), it is defined.
print(f"coverage@10, ties broken by identifier: {coverage(low_ids_first, train, k=10):.4f}")
print(f"coverage of all the candidates:         {coverage(scored, train):.4f}")
