Quickstart
==========

.. contents:: Table of Contents

To use recsyslearn in a project:

.. code-block:: python

    import recsyslearn

To use its features in a project you just need a dataset
in the form of ``user, item, rating`` or
``user, item, timestamp``.

Segment your dataset
---------------------

Before evaluating the fairness of your recommendation system, you may
want to segment your dataset in groups of users or items based on one
of their features.

To segment your dataset in groups (e.g. based on the item popularity):

.. code-block:: python

    import json
    import pandas as pd
    from recsyslearn.dataset.segmentations import InteractionSegmentation

    # Read the entire dataset (in the form user, item, rank)
    train_data = pd.read_csv("train_dataset.csv")

    # Segment the dataset in two groups based on the item popularity
    segmented_items = InteractionSegmentation().segment(train_data, [0.8, 0.2])

    # Print the results
    print(segmented_items.head())


Evaluate the Accuracy
---------------------

To evaluate the accuracy of your recommendation system:

.. code-block:: python

    import json
    import pandas as pd
    from recsyslearn.accuracy.metrics import NDCG
    from recsyslearn.dataset.utils import find_relevant_items

    # Read the recommendation lists (in the form user, item, rank)
    top_k = pd.read_csv("top_k.csv")

    # Read the test dataset against you would like to evaluate accuracy
    # (in the form user, item, rank)
    dataset = pd.read_csv("dataset.csv")

    # Find the relevant items for each user in the test dataset
    pos_items = find_relevant_items(dataset)

    # Evaluate the accuracy with NDCG@5 and NDCG@10
    ats = (5, 10)
    ndcg_df = NDCG().evaluate(top_k, pos_items, ats)

    # Print the results
    print(json.dumps({f"NDCG@{at}": ndcg_df[f"NDCG@{at}"].mean() for at in ats}, indent=4))


Evaluate the Beyond Accuracy
----------------------------

To evaluate the Beyond Accuracy (e.g. Novelty) of your recommendation system:

.. code-block:: python

    import json
    import pandas as pd
    from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
    from recsyslearn.dataset.segmentations import InteractionSegmentation

    # Read the entire dataset (in the form user, item, rank)
    train_data = pd.read_csv("train_dataset.csv")

    # Segment the dataset in two groups based on the item popularity
    segmented_items = InteractionSegmentation().segment(train_data, [0.8, 0.2])

    # Read the recommendation lists (in the form user, item, rank)
    top_k = pd.read_csv("top_k.csv")

    # Merge the recommendation lists with the item groups
    top_k_with_item_groups = top_k.merge(segmented_items, on="item")

    # Evaluate the Novelty
    novelty = Novelty().evaluate(top_k_with_item_groups)

    # Print the results
    print(json.dumps({"novelty": novelty}, indent=4))

Evaluate the Fairness
---------------------

To evaluate the fairness (e.g. Kullback-Leibler Divergence) of your recommendation system:

.. code-block:: python

    import json
    import pandas as pd
    from recsyslearn.dataset.segmentations import (
        ActivitySegmentation,
        InteractionSegmentation,
    )
    from recsyslearn.fairness.metrics import KullbackLeibler

    # Read the entire dataset (in the form user, item, rank)
    train_data = pd.read_csv("train_dataset.csv")

    # Segment the dataset in two groups based on the item popularity
    segmented_items = InteractionSegmentation().segment(train_data, [0.8, 0.2])

    # Read the recommendation lists (in the form user, item, rank)
    top_k = pd.read_csv("top_k.csv")

    # Merge the recommendation lists with the item groups
    top_k_with_item_groups = top_k.merge(segmented_items, on="item")

    # Read the test dataset against you would like to evaluate accuracy
    # (in the form user, item, rank)
    test_data = pd.read_csv("test_dataset.csv")

    # Set the target representation of the item groups
    target_representation = pd.DataFrame(
        [["1", 0.5], ["2", 0.5]], columns=["group", "target_representation"]
    )

    # Evaluate the Kullback-Leibler Divergence
    divergence = KullbackLeibler().evaluate(top_k_with_item_groups, target_representation)

    # Print the results
    print(json.dumps({"KL@[0.5, 0.5]": divergence}, indent=4))
