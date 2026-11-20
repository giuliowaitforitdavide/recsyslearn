"""Beyond-accuracy qualities of a recommender: catalog coverage and novelty."""

import pandas as pd

from recsyslearn.beyond_accuracy import coverage, novelty

train_data = pd.read_csv("./examples/__assets__/train_dataset.csv")
top_k = pd.read_csv("./examples/__assets__/top_k.csv")

# Share of the items of the training set that are recommended to at least one user.
print(f"coverage@10: {coverage(top_k, train_data, k=10):.4f}")

# Mean self-information (in bits) of the recommended items: the popularity of an item
# is the fraction of training users who interacted with it.
print(f"novelty@10:  {novelty(top_k, train_data, k=10):.4f}")
