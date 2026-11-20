Quickstart
==========

The examples below use three tables in the format described in :doc:`data`: the
training interactions, the held-out test interactions, and the top-10 recommendations
of a model. The files are in the ``examples/__assets__`` folder of the repository.

.. code-block:: python

    import pandas as pd

    # Training and held-out interactions: user, item.
    train = pd.read_csv("examples/__assets__/train_dataset.csv")
    test = pd.read_csv("examples/__assets__/test_dataset.csv")
    # Recommendations: user, item, rank.
    top_k = pd.read_csv("examples/__assets__/top_k.csv")

The interactions are implicit feedback. With explicit feedback, a ``rating`` column in
the ground truth is the graded relevance of each pair.

Accuracy
--------

.. code-block:: python

    from recsyslearn.accuracy import ndcg

    ndcg(top_k, test, k=10)                  # mean over the test users
    ndcg(top_k, test, k=10, per_user=True)   # one score per user

Scores instead of ranks
-----------------------

Lists can be ordered by a ``score`` column (the highest is the best) instead of a
``rank``. Scores can be tied, and then no order is chosen for the tied items: see
:doc:`data`.

.. code-block:: python

    # The same lists, so the same value.
    scored = top_k.assign(score=1 / top_k["rank"]).drop(columns="rank")
    ndcg(scored, test, k=10)

Beyond accuracy
---------------

.. code-block:: python

    from recsyslearn.beyond_accuracy import coverage, novelty

    # Share of the training items that get recommended.
    coverage(top_k, train, k=10)
    # Mean self-information of the recommended items, in bits.
    novelty(top_k, train, k=10)

Segment items and users
-----------------------

Segmentations split items or users into groups, typically on the training set.

.. code-block:: python

    from recsyslearn import segmentation

    # Short head (items that account for 80% of the interactions) and long tail.
    item_groups = segmentation.by_cumulative_interactions(
        train, (0.8, 0.2), labels=("head", "tail")
    )

    # The 20% most active users and the others.
    user_groups = segmentation.by_interaction_count(
        train, (0.2, 0.8), by="user", labels=("active", "casual")
    )

    # Popularity of every item as the share of interactions it received.
    segmentation.popularity(train)

    # Groups from any categorical attribute, e.g. the gender of the users.
    # segmentation.by_feature(users, "gender")

The segmentations return a table ``[item | user, "group"]``, which the fairness metrics
take as it is; ``popularity`` returns ``[item | user, "popularity"]``.

Fairness
--------

The fairness metrics describe how the *benefit* of being recommended is distributed
over the groups. By default the benefit is the exposure, which decays with the position
in the list; pass ``ground_truth=test`` to measure the effectiveness instead, i.e. to
count only relevant recommendations.

.. code-block:: python

    from recsyslearn import fairness

    # Share of the exposure received by head and tail items.
    fairness.group_distribution(top_k, item_groups)

    # How far is it from a target? Here: proportional to the size of each group.
    target = item_groups["group"].value_counts(normalize=True)
    fairness.kl_divergence(top_k, item_groups, target)

    # How evenly is it spread? (1 bit = evenly over two groups)
    fairness.entropy(top_k, item_groups)

    # Are relevant recommendations spread over user groups like the users are?
    target = user_groups["group"].value_counts(normalize=True)
    fairness.kl_divergence(top_k, user_groups, target, ground_truth=test)

    # Do active and casual users get different mixes of head and tail items?
    fairness.mutual_information(
        top_k, user_groups=user_groups, item_groups=item_groups
    )

The :doc:`../methodology` page lists the formula and the source of every function.
:doc:`examples` has a complete script for each of these topics, and for tied scores, the
comparison of several recommenders and the errors.
