"""Beyond accuracy: coverage of the catalog and novelty of the lists."""

import pandas as pd

from recsyslearn.beyond_accuracy import coverage, novelty

train = pd.read_csv("./examples/__assets__/train_dataset.csv")
top_k = pd.read_csv("./examples/__assets__/top_k.csv")

# Coverage: the share of the catalog recommended to at least one user. The
# catalog is what could be recommended, and it is passed explicitly: here,
# the items of the training set (any table with an `item` column works).
for k in (1, 5, 10):
    print(f"coverage@{k}: {coverage(top_k, train, k=k):.4f}")

# The catalog can also be any collection of item identifiers: a list, a set,
# an array, an index. A larger catalog (items that exist but have no
# interaction) lowers the share.
items = train["item"].unique()
cold = range(int(items.max()) + 1, int(items.max()) + 501)
value = coverage(top_k, [*items, *cold], k=10)
print(f"coverage@10 with 500 more items in the catalog: {value:.4f}")

# Novelty: the mean self-information, in bits, of the recommended items. The
# popularity of an item is the fraction of the users of `interactions` who
# interacted with it: an item known by half of the users carries 1 bit, one
# known by a quarter 2 bits.
for k in (1, 5, 10):
    print(f"novelty@{k}: {novelty(top_k, train, k=k):.4f} bits")

# One value per user: who gets the least popular recommendations?
per_user = novelty(top_k, train, k=10, per_user=True)
value = per_user.nlargest(3).round(3).to_string()
print(f"\nmost novel lists (user, bits):\n{value}")

# A recommended item that nobody interacted with has no popularity: it is
# given the self-information of an item with a single user, log2(number of
# users), the maximum.
unseen = pd.DataFrame(
    {"user": [1], "item": [int(items.max()) + 1], "rank": [1]}
)
value = novelty(unseen, train)
print(f"\nnovelty of an item without interactions: {value:.4f} bits")
