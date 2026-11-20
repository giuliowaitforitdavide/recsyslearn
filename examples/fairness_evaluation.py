"""Group fairness of a recommender: how exposure is spread over groups."""

import pandas as pd

from recsyslearn import fairness, segmentation

train_data = pd.read_csv("./examples/__assets__/train_dataset.csv")
test_data = pd.read_csv("./examples/__assets__/test_dataset.csv")
top_k = pd.read_csv("./examples/__assets__/top_k.csv")

# Items: short head (80% of the training interactions) vs long tail.
item_groups = segmentation.by_cumulative_interactions(
    train_data, (0.8, 0.2), labels=("head", "tail")
)
# Users: the 20% most active vs the others.
user_groups = segmentation.by_interaction_count(
    train_data, (0.2, 0.8), by="user", labels=("active", "casual")
)

print("Exposure of the item groups:")
print(fairness.group_distribution(top_k, item_groups).to_string())

# Item fairness: is exposure proportional to the size of each group in the catalog?
catalog_share = item_groups["group"].value_counts(normalize=True)
print(f"KL from the catalog share: {fairness.kl_divergence(top_k, item_groups, catalog_share):.4f}")
print(f"KL from parity:            {fairness.kl_divergence(top_k, item_groups):.4f}")
print(f"entropy:                   {fairness.entropy(top_k, item_groups):.4f}")

# User fairness: are relevant recommendations (effectiveness) spread like the users?
user_share = user_groups["group"].value_counts(normalize=True)
divergence = fairness.kl_divergence(top_k, user_groups, user_share, ground_truth=test_data)
print(f"KL of effectiveness from the user share: {divergence:.4f}")

# Dependence: do the two user groups get different mixes of head and tail items?
dependence = fairness.mutual_information(top_k, user_groups=user_groups, item_groups=item_groups)
print(f"mutual information: {dependence:.4f}")
