from recsyslearn.fairness.metrics import *
from recsyslearn.dataset.utils import find_relevant_items
from recsyslearn.dataset.segmentations import *
import pandas as pd
import json

train_data = pd.read_csv(
    './examples/__assets__/train_dataset.csv')
segmented_users = ActivitySegmentation().segment(
    train_data, [0.2, 0.8], min_interaction=10)
segmented_items = InteractionSegmentation().segment(
    train_data, [0.8, 0.2], min_interaction=10)

segmented_items['item'] = segmented_items['item'].to_string()
segmented_items['group'] = segmented_items['group'].to_string()


top_k = pd.read_csv('./examples/__assets__/top_k.csv')
top_k_with_item_groups = top_k.merge(segmented_items, on='item')
test_data = pd.read_csv('./examples/__assets__/test_dataset.csv')
pos_items = find_relevant_items(test_data)
target_representation = pd.DataFrame([['1', 0.5], ['2', 0.5]], columns=[
    'group', 'target_representation'])
divergence = KullbackLeibler().evaluate(
    top_k_with_item_groups, target_representation, pos_items)

print(
    json.dumps({'KL@[0.5, 0.5]': divergence}, indent=4)
)
