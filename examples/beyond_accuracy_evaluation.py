import json

import pandas as pd

from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from recsyslearn.dataset.segmentations import InteractionSegmentation

train_data = pd.read_csv("./examples/__assets__/train_dataset.csv")
segmented_items = InteractionSegmentation().segment(train_data, [0.8, 0.2])

top_k = pd.read_csv("./examples/__assets__/top_k.csv")
top_k = top_k.astype({"item": str, "user": str, "rank": int})
top_k_with_item_groups = top_k.merge(segmented_items, on="item")
coverage = Coverage().evaluate(
    top_k_with_item_groups, segmented_items["item"].to_list()
)
novelty = Novelty().evaluate(top_k_with_item_groups)

print(json.dumps({"coverage": coverage, "Novelty": novelty}, indent=4))
