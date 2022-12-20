from recsyslearn.dataset.segmentations import *
import pandas as pd
import json

train_data = pd.read_csv(
    './examples/__assets__/train_dataset.csv')
segmented_users = ActivitySegmentation().segment(
    train_data, [0.2, 0.8], min_interaction=10)
segmented_items = InteractionSegmentation().segment(
    train_data, [0.8, 0.2], min_interaction=10)

print(
    json.dumps({
        'user_groups': {
            "1": str(segmented_users[segmented_users.group == "1"].size),
            "2": str(segmented_users[segmented_users.group == "2"].size)
        },
        'item_groups': {
            "1": str(segmented_items[segmented_items.group == "1"].size),
            "2": str(segmented_items[segmented_items.group == "2"].size)
        }
    }, indent=4)
)
