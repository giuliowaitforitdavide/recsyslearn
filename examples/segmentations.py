"""Segmentations of items and users."""

import pandas as pd

from recsyslearn import segmentation

train_data = pd.read_csv("./examples/__assets__/train_dataset.csv")

# Short head / long tail: the head items account for 80% of the interactions.
head_tail = segmentation.by_cumulative_interactions(train_data, (0.8, 0.2), labels=("head", "tail"))
print(head_tail["group"].value_counts().to_string(), end="\n\n")

# Head / mid / tail, 20% / 60% / 20% of the interactions.
head_mid_tail = segmentation.by_cumulative_interactions(
    train_data, (0.2, 0.6, 0.2), labels=("head", "mid", "tail")
)
print(head_mid_tail["group"].value_counts().to_string(), end="\n\n")

# The 5% most active users against everybody else.
activity = segmentation.by_interaction_count(
    train_data, (0.05, 0.95), by="user", labels=("advantaged", "disadvantaged")
)
print(activity["group"].value_counts().to_string(), end="\n\n")

# Popularity of every item as the share of interactions it received.
print(segmentation.popularity(train_data).nlargest(5, "popularity").to_string(index=False))

# Any categorical attribute works as a segmentation, e.g. a (made up) parity of the id.
features = pd.DataFrame({"item": train_data["item"].unique()})
features["parity"] = features["item"].mod(2).map({0: "even", 1: "odd"})
print(segmentation.by_feature(features, "parity")["group"].value_counts().to_string())
