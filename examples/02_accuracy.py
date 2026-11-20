"""Ranking accuracy: nDCG@k on implicit and on explicit feedback."""

import numpy as np
import pandas as pd

from recsyslearn.accuracy import ndcg

# Top-10 lists of a model for 943 users (user, item, rank) and one held-out interaction
# per user (user, item): implicit feedback.
top_k = pd.read_csv("./examples/__assets__/top_k.csv")
test = pd.read_csv("./examples/__assets__/test_dataset.csv")

# The cutoff: only the first k items of every list are evaluated. None is the whole list.
for k in (1, 5, 10, None):
    print(f"nDCG@{k}: {ndcg(top_k, test, k=k):.4f}")

# One score per user instead of the mean, e.g. for a significance test or to compare
# groups of users. The result is a Series indexed by user.
per_user = ndcg(top_k, test, k=10, per_user=True)
print(f"\nusers with a hit in the top 10: {(per_user > 0).mean():.1%}")
print(per_user.describe().loc[["mean", "std", "max"]].to_string())

# A bootstrap confidence interval of the mean, from the per-user scores.
rng = np.random.default_rng(0)
means = rng.choice(per_user.to_numpy(), size=(1000, len(per_user))).mean(axis=1)
low, high = np.percentile(means, [2.5, 97.5])
print(f"95% bootstrap interval of nDCG@10: [{low:.4f}, {high:.4f}]")

# Who is evaluated: the users with at least one relevant item. A user of the ground
# truth who received no recommendation scores 0; a user with recommendations but no
# relevant item is left out, because the metric is not defined for them.
half = top_k[top_k["user"] % 2 == 0]
evaluated = len(ndcg(half, test, k=10, per_user=True))
print(f"\nlists for half of the users:        nDCG@10 = {ndcg(half, test, k=10):.4f}")
print(f"  users evaluated: {evaluated} of {test['user'].nunique()} (those without a list score 0)")
few = test[test["user"] <= 100]
evaluated = len(ndcg(top_k, few, k=10, per_user=True))
print(f"ground truth for the first 100 users: nDCG@10 = {ndcg(top_k, few, k=10):.4f}")
print(f"  users evaluated: {evaluated} of {top_k['user'].nunique()} (the others are left out)")

# Explicit feedback: a `rating` column in the ground truth is the graded relevance. The
# files hold implicit feedback, so here a rating from 1 to 5 is made up for each pair.
rated = test.assign(rating=rng.integers(1, 6, size=len(test)).astype(float))
# With one relevant item per user the grade cancels out in DCG / IDCG, so the example
# also gives every user a second relevant item: the first one of their list, graded 1.
first = top_k.loc[top_k["rank"] == 1, ["user", "item"]].assign(rating=1.0)
rated = pd.concat([rated, first]).drop_duplicates(["user", "item"])
# The gain turns a grade into a contribution: linear (the grade itself) or exponential
# (2^grade - 1), which rewards the highest grades much more.
for gain in ("linear", "exponential"):
    print(f"graded nDCG@10, {gain:<11} gain: {ndcg(top_k, rated, k=10, gain=gain):.4f}")
