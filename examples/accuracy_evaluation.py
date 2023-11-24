import json

import pandas as pd

from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.dataset.utils import find_relevant_items

top_k = pd.read_csv("./examples/__assets__/top_k.csv")
test_data = pd.read_csv("./examples/__assets__/test_dataset.csv")
pos_items = find_relevant_items(test_data)
ats = (5, 10)
ndcg_df = NDCG().evaluate(top_k, pos_items, ats)


print(json.dumps({f"NDCG@{at}": ndcg_df[f"NDCG@{at}"].mean() for at in ats}, indent=4))
