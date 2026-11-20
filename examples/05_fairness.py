"""Group fairness: how the benefit of recommendations is spread."""

import pandas as pd

from recsyslearn import fairness, segmentation

train = pd.read_csv("./examples/__assets__/train_dataset.csv")
test = pd.read_csv("./examples/__assets__/test_dataset.csv")
top_k = pd.read_csv("./examples/__assets__/top_k.csv")

item_groups = segmentation.by_cumulative_interactions(
    train, (0.8, 0.2), labels=("head", "tail")
)
user_groups = segmentation.by_interaction_count(
    train, (0.2, 0.8), by="user", labels=("active", "casual")
)

# 1. The distribution every other metric is computed on: the share of the
#    benefit that each group receives. By default the benefit is the
#    exposure, which decays with the position (1 / log2(1 + position)).
print("Exposure of the item groups:")
print(fairness.group_distribution(top_k, item_groups).round(4).to_string())

# With flat exposure every recommendation counts the same: the share of the
# slots.
flat = fairness.group_distribution(top_k, item_groups, exposure="flat")
print(f"share of the slots given to the tail:           {flat['tail']:.4f}")

# A cutoff, as in every other metric.
top_1 = fairness.group_distribution(top_k, item_groups, k=1)
print(f"share of the first positions given to the tail: {top_1['tail']:.4f}")

# 2. Groups of users instead of groups of items: who receives the benefit.
#    With a ground truth the benefit is the effectiveness: only relevant
#    recommendations count.
print("\nExposure and effectiveness of the user groups:")
exposure = fairness.group_distribution(top_k, user_groups)
effectiveness = fairness.group_distribution(
    top_k, user_groups, ground_truth=test
)
print(
    pd.DataFrame({"exposure": exposure, "effectiveness": effectiveness})
    .round(4)
    .to_string()
)

# 3. Entropy: how evenly the benefit is spread. In bits, from 0 (one group
#    takes everything) to log2(number of groups); normalised, from 0 to 1.
three = segmentation.by_cumulative_interactions(
    train, (0.2, 0.6, 0.2), labels=("head", "mid", "tail")
)
value = fairness.entropy(top_k, three)
print(f"\nentropy over head, mid and tail: {value:.4f} bits of 1.5850")
value = fairness.entropy(top_k, three, normalize=True)
print(f"normalised entropy:              {value:.4f}")

# 4. Kullback-Leibler divergence from the distribution regarded as fair: 0
#    when the benefit is spread exactly like the target. Without a target,
#    the target is parity.
value = fairness.kl_divergence(top_k, item_groups)
print(f"\nKL from parity:                 {value:.4f}")

# The share of the catalog of each group, as a Series indexed by group.
catalog_share = item_groups["group"].value_counts(normalize=True)
divergence = fairness.kl_divergence(top_k, item_groups, catalog_share)
print(f"KL from the catalog share:      {divergence:.4f}")

# The share of the training interactions, as a table [group, probability].
interactions = train.merge(item_groups, on="item")["group"].value_counts(
    normalize=True
)
target = (
    interactions.rename("probability").rename_axis("group").reset_index()
)
value = fairness.kl_divergence(top_k, item_groups, target)
print(f"KL from the interaction share:  {value:.4f}")

# Any other target, as a dictionary.
target = {"head": 0.6, "tail": 0.4}
value = fairness.kl_divergence(top_k, item_groups, target)
print(f"KL from a 60 / 40 target:       {value:.4f}")

# User fairness: are relevant recommendations spread like the users are?
user_share = user_groups["group"].value_counts(normalize=True)
divergence = fairness.kl_divergence(
    top_k, user_groups, user_share, ground_truth=test
)
print(f"KL of effectiveness from the user share: {divergence:.4f}")

# 5. Mutual information: do different groups of users get different groups of
#    items? 0 when what a user is shown does not depend on their group.
dependence = fairness.mutual_information(
    top_k, user_groups=user_groups, item_groups=item_groups
)
print(
    f"\nmutual information, groups of users and of items: {dependence:.4f}"
)

# Leaving one side out makes every user (or item) a group of its own.
individual = fairness.mutual_information(top_k, item_groups=item_groups)
print(f"mutual information, single users and item groups: {individual:.4f}")

# 6. Groups from an attribute instead of from the interactions. Every
#    recommended item needs a group; items that are never recommended do not.
features = pd.DataFrame({"item": top_k["item"].unique()})
features["parity"] = (
    features["item"].mod(2).map({0: "even", 1: "odd"})
)  # made up
by_parity = segmentation.by_feature(features, "parity")
print("\nExposure by a (made up) attribute:")
print(fairness.group_distribution(top_k, by_parity).round(4).to_string())
