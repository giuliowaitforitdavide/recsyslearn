"""The tables recsyslearn reads: the columns and the order of the lists.

Everything is built by hand, so that every number can be checked on paper.
"""

import math

import pandas as pd

from recsyslearn.accuracy import ndcg

# Held-out interactions of two users. With the columns user and item only,
# the feedback is implicit: every listed pair is relevant.
truth = pd.DataFrame(
    {"user": ["ann", "ann", "bob"], "item": ["x", "z", "y"]}
)

# 1. Recommendations ordered by `rank`: 1 is the best item of the user.
by_rank = pd.DataFrame(
    {
        "user": ["ann", "ann", "ann", "bob", "bob", "bob"],
        "item": ["x", "y", "z", "x", "y", "z"],
        "rank": [1, 2, 3, 1, 2, 3],
    }
)
# ann: hits at positions 1 and 3, the ideal list has them at 1 and 2.
ann = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
# bob: one hit at position 2, the ideal list has it at 1.
bob = 1 / math.log2(3)
value = ndcg(by_rank, truth)
print(f"by rank:           {value:.4f}  (by hand: {(ann + bob) / 2:.4f})")

# 2. Only the order of the ranks matters, not their values: zero-based ranks,
#    ranks with gaps and fractional ranks give the same lists. The order of
#    the rows is irrelevant.
shuffled = by_rank.assign(rank=[0, 10, 25.5, 0, 10, 25.5]).sample(
    frac=1, random_state=0
)
print(f"any scale of rank: {ndcg(shuffled, truth):.4f}")

# 3. Recommendations ordered by `score`: the largest score is the best item.
#    This is what most models and tools produce.
by_score = by_rank.drop(columns="rank").assign(
    score=[0.9, 0.5, 0.1, 7.0, 3.0, -2.0]
)
print(f"by score:          {ndcg(by_score, truth):.4f}")

# 4. With neither column, the order of the rows is the ranking.
by_rows = by_rank[["user", "item"]]
print(f"by order of rows:  {ndcg(by_rows, truth):.4f}")

# 5. With both columns, `rank` is used and `score` is not read.
both = by_rank.assign(score=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print(f"rank and score:    {ndcg(both, truth):.4f}")

# 6. Identifiers can be of any hashable type (integers, strings, categories),
#    as long as the tables that must agree use the same one. Columns that a
#    function does not read are ignored, and so is the index.
codes = {"ann": 1, "bob": 2, "x": 10, "y": 20, "z": 30}
numbered = (
    by_rank.replace(codes).assign(model="demo").set_index("rank", drop=False)
)
print(f"integer ids:       {ndcg(numbered, truth.replace(codes)):.4f}")

# 7. Explicit feedback: a `rating` column in the ground truth is the graded
#    relevance.
rated = truth.assign(rating=[1.0, 3.0, 2.0])
print(f"graded relevance:  {ndcg(by_rank, rated):.4f}")

# 8. No function modifies its arguments.
before = by_rank.copy()
ndcg(by_rank, truth, k=2)
print(f"input unchanged:   {by_rank.equals(before)}")
