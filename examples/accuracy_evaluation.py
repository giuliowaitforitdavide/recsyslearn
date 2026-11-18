import json

import pandas as pd

from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.dataset.utils import find_relevant_items

top_k = pd.read_csv("./examples/__assets__/top_k.csv")
test_data = pd.read_csv("./examples/__assets__/test_dataset.csv")
pos_items = find_relevant_items(test_data)
results = [NDCG(k=k).evaluate(top_k, pos_items) for k in (5, 10)]

print(json.dumps({result.name: result.aggregate for result in results}, indent=4))
