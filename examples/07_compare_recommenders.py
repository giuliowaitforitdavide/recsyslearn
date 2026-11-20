"""Compare recommenders on every metric at once.

Three sets of top-10 lists for the same users: the model of the other examples, a
popularity baseline and a random one. Accuracy alone would hide how differently they
treat the catalog.
"""

import numpy as np
import pandas as pd

from recsyslearn import fairness, segmentation
from recsyslearn.accuracy import ndcg
from recsyslearn.beyond_accuracy import coverage, novelty

K = 10
train = pd.read_csv("./examples/__assets__/train_dataset.csv")
test = pd.read_csv("./examples/__assets__/test_dataset.csv")
model = pd.read_csv("./examples/__assets__/top_k.csv")

users = pd.DataFrame({"user": train["user"].unique()})
counts = train["item"].value_counts().rename("score").rename_axis("item").reset_index()


def top_unseen(candidates: pd.DataFrame) -> pd.DataFrame:
    """The K best-scored candidates of every user that are not in the training set."""
    merged = candidates.merge(train, on=["user", "item"], how="left", indicator=True)
    unseen = merged[merged["_merge"] == "left_only"]
    # Ties broken by identifier, to have complete lists with a definite order.
    unseen = unseen.sort_values(["user", "score", "item"], ascending=[True, False, True])
    unseen = unseen.assign(rank=unseen.groupby("user").cumcount() + 1)
    return unseen.loc[unseen["rank"] <= K, ["user", "item", "rank"]]


most_popular = top_unseen(users.merge(counts.nlargest(150, "score", keep="all"), how="cross"))
rng = np.random.default_rng(0)
everything = users.merge(counts[["item"]], how="cross")
random = top_unseen(everything.assign(score=rng.random(len(everything))))

item_groups = segmentation.by_cumulative_interactions(train, (0.8, 0.2), labels=("head", "tail"))
user_groups = segmentation.by_interaction_count(
    train, (0.2, 0.8), by="user", labels=("active", "casual")
)
catalog_share = item_groups["group"].value_counts(normalize=True)
user_share = user_groups["group"].value_counts(normalize=True)


def evaluate(recommendations: pd.DataFrame) -> dict[str, float]:
    """Every metric of the library on one set of lists."""
    return {
        "nDCG": ndcg(recommendations, test, k=K),
        "coverage": coverage(recommendations, train, k=K),
        "novelty": novelty(recommendations, train, k=K),
        "tail exposure": fairness.group_distribution(recommendations, item_groups, k=K)["tail"],
        "item KL": fairness.kl_divergence(recommendations, item_groups, catalog_share, k=K),
        "user KL": fairness.kl_divergence(
            recommendations, user_groups, user_share, k=K, ground_truth=test
        ),
        "dependence": fairness.mutual_information(
            recommendations, user_groups=user_groups, item_groups=item_groups, k=K
        ),
    }


recommenders = {"model": model, "most popular": most_popular, "random": random}
table = pd.DataFrame({name: evaluate(lists) for name, lists in recommenders.items()})
print(f"Metrics at {K} (item KL: exposure against the catalog share of head and tail;")
print("user KL: effectiveness against the share of active and casual users)\n")
print(table.round(4).to_string())
