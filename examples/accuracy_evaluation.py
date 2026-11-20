"""Ranking accuracy of a recommender: nDCG@k."""

import pandas as pd

from recsyslearn.accuracy import ndcg

# Recommendation lists: user, item, rank (1 = best).
top_k = pd.read_csv("./examples/__assets__/top_k.csv")
# Held-out interactions: user, item, rank (the feedback value).
test_data = pd.read_csv("./examples/__assets__/test_dataset.csv")

for k in (5, 10):
    print(f"nDCG@{k}: {ndcg(top_k, test_data, k=k):.4f}")

# One score per user, e.g. to run a significance test or to compare user groups.
per_user = ndcg(top_k, test_data, k=10, per_user=True)
print(per_user.describe())
