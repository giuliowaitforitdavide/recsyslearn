"""Segmentations: split items or users into groups to study separately."""

import pandas as pd

from recsyslearn import segmentation

train = pd.read_csv("./examples/__assets__/train_dataset.csv")


def sizes(groups: pd.DataFrame) -> str:
    """Number of members of every group, on one line."""
    return ", ".join(
        f"{group}: {size}"
        for group, size in groups["group"].value_counts().items()
    )


# By share of the interactions. Short head / long tail: the head is the
# smallest set of most popular items that accounts for 80% of the
# interactions.
head_tail = segmentation.by_cumulative_interactions(
    train, (0.8, 0.2), labels=("head", "tail")
)
print(f"head / tail items:        {sizes(head_tail)}")

# Any number of groups: head, mid and tail with 20%, 60% and 20% of the
# interactions.
three = segmentation.by_cumulative_interactions(
    train, (0.2, 0.6, 0.2), labels=("head", "mid", "tail")
)
print(f"head / mid / tail items:  {sizes(three)}")

# Without labels the groups are numbered from 1, the most popular.
value = sizes(segmentation.by_cumulative_interactions(train))
print(f"default labels:           {value}")

# The same for users: the most active users who make half of the
# interactions.
heavy = segmentation.by_cumulative_interactions(
    train, (0.5, 0.5), by="user", labels=("heavy", "light")
)
print(f"heavy / light users:      {sizes(heavy)}")

# By share of the population: here the proportions say how many items (users)
# are in each group, not how many interactions they account for.
popular = segmentation.by_interaction_count(
    train, (0.2, 0.8), labels=("popular", "niche")
)
print(f"20% most popular items:   {sizes(popular)}")
active = segmentation.by_interaction_count(
    train, (0.05, 0.95), by="user", labels=("advantaged", "disadvantaged")
)
print(f"5% most active users:     {sizes(active)}")

# Popularity as a number instead of a group. Relative to the interactions it
# is the share of all the interactions (it sums to 1); relative to the
# population it is the fraction of the users who interacted with the item.
share = segmentation.popularity(train)
reach = segmentation.popularity(train, relative_to="population")
top = share.merge(reach, on="item", suffixes=("_share", "_reach")).nlargest(
    3, "popularity_share"
)
print(f"\n{top.round(4).to_string(index=False)}")
print(f"sum of the shares: {share['popularity'].sum():.4f}")

# The activity of the users, with the same function.
activity = segmentation.popularity(train, by="user")
value = activity["popularity"].max()
print(f"most active user makes {value:.2%} of the interactions")

# By a categorical attribute: any column of a table with one row per item (or
# user). A missing value is not dropped: it becomes a group of its own.
features = pd.DataFrame({"item": train["item"].unique()})
# A made-up attribute, missing for a quarter of the items.
features["decade"] = (
    features["item"].mod(4).map({0: "1970s", 1: "1980s", 2: "1990s"})
)
value = sizes(segmentation.by_feature(features, "decade"))
print(f"\nby a feature:             {value}")
by_decade = segmentation.by_feature(features, "decade", missing="not known")
print(f"label of the missing:     {sizes(by_decade)}")

# Every segmentation returns the columns [item | user, group]: the table the
# fairness metrics take.
print(f"\n{head_tail.head(3).to_string(index=False)}")
