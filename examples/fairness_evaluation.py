import json

import pandas as pd

from recsyslearn.dataset.segmentations import (
    ActivitySegmentation,
    InteractionSegmentation,
)
from recsyslearn.fairness.metrics import KullbackLeibler, MutualInformation

train_data = pd.read_csv("./examples/__assets__/train_dataset.csv")
segmented_users = ActivitySegmentation().segment(train_data, [0.2, 0.8])
segmented_items = InteractionSegmentation().segment(train_data, [0.8, 0.2])

top_k = pd.read_csv("./examples/__assets__/top_k.csv")
top_k = top_k.astype({"item": str, "user": str, "rank": int})
top_k_with_item_groups = top_k.merge(segmented_items, on="item")
top_k_with_user_groups = top_k.merge(segmented_users, on="user")
test_data = pd.read_csv("./examples/__assets__/test_dataset.csv")
target_representation = pd.DataFrame(
    [["1", 0.5], ["2", 0.5]], columns=["group", "target_representation"]
)


divergence = KullbackLeibler().evaluate(top_k_with_item_groups, target_representation)
mi = MutualInformation().evaluate(top_k_with_user_groups, "user")

print(json.dumps({"KL@[0.5, 0.5]": divergence, "Users MI": mi}, indent=4))
