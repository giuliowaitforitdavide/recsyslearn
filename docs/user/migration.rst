Migrating from version 3
========================

Version 4 is a rewrite: the metrics were re-derived from the literature, the core is
written in Rust, and the class-based API was replaced by plain functions. Results can
differ from version 3 where version 3 departed from the published definitions; those
cases are marked below.

Installing
----------

* Python 3.10 is no longer supported: version 4 needs Python 3.11 or later.
* The package is distributed as compiled wheels. Where no wheel is available, pip builds
  it from source, which needs a Rust toolchain: see :doc:`install`.

Modules
-------

The functions are in four modules, imported as
``from recsyslearn import accuracy, beyond_accuracy, fairness, segmentation``. The
modules of version 3 no longer exist:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Version 3
     - Version 4
   * - ``recsyslearn.accuracy.metrics``
     - ``recsyslearn.accuracy``
   * - ``recsyslearn.beyond_accuracy.metrics``
     - ``recsyslearn.beyond_accuracy``
   * - ``recsyslearn.fairness.metrics``
     - ``recsyslearn.fairness``
   * - ``recsyslearn.dataset``, ``recsyslearn.dataset.segmentations``
     - ``recsyslearn.segmentation``
   * - ``recsyslearn.errors.errors``
     - ``recsyslearn.errors``
   * - ``recsyslearn.fairness.utils``, ``recsyslearn.utils``,
       ``recsyslearn.dataset.utils``
     - Removed, see below.

Classes and functions
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Version 3
     - Version 4
   * - ``NDCG().evaluate(top_n, find_relevant_items(test), ats)``
     - ``accuracy.ndcg(top_n, test, k=...)``, one call per cutoff. Version 3 returned a
       table with a row per user and a column ``NDCG@k`` per cutoff; version 4 returns
       the mean over the users, and ``per_user=True`` returns the scores as a Series
       indexed by user. The feedback value of ``test`` is now read from a ``rating``
       column and used as graded relevance: leave the column out to keep binary
       relevance. **Different values**: a user of ``test`` without recommendations
       scores 0, where version 3 left them out; and the lists are ordered by ``rank``,
       where version 3 read the rows in the order of the table. A cutoff beyond the
       length of the lists is accepted (version 3 raised
       ``RecListTooShortException``).
   * - ``find_relevant_items``
     - Removed: pass the test interactions directly.
   * - ``Coverage().evaluate(top_n, items)``
     - ``beyond_accuracy.coverage(top_n, items, k=...)``. An identifier repeated in
       ``items`` counts once, and a recommended item that is not in ``items`` is
       refused; version 3 divided the number of recommended items by ``len(items)``
       whatever they were.
   * - ``Novelty().evaluate(top_n_with_popularity, popularity_definition=...)``
     - ``beyond_accuracy.novelty(top_n, train, k=...)``. **Different values**: the
       popularity is computed from the interactions as the fraction of users who
       interacted with the item, instead of being read from the column named by
       ``popularity_definition``, which is removed.
   * - ``Entropy().evaluate(top_n_with_groups)``
     - ``fairness.entropy(top_n, groups)``. **Different values**: version 3 weighted
       recommendations by their raw rank; version 4 uses the exposure
       ``1 / log2(1 + position)`` (or ``exposure="flat"``).
   * - ``KullbackLeibler().evaluate(top_n_with_groups, target_representation)``
     - ``fairness.kl_divergence(top_n, groups, target)``, where ``target`` maps each
       group to its probability and defaults to parity.
   * - ``MutualInformation().evaluate(top_n_with_groups, "user")``
     - ``fairness.mutual_information(top_n, user_groups=groups)``; with ``"item"``,
       ``item_groups=groups``. Both can now be given at once. An invalid side is no
       longer a ``KeyError``.
   * - ``exp_matrix``, ``prob_matrix``, ``eff_matrix``
     - Removed. ``fairness.group_distribution(top_n, groups)`` returns the probability
       of each group, which is what the three functions were combined to obtain.
   * - ``check_columns_exist``
     - Removed: every function checks its own tables.
   * - ``rel_matrix=...``
     - ``ground_truth=test``.
   * - ``InteractionSegmentation().segment(data, proportions, group=...)``
     - ``segmentation.by_cumulative_interactions(data, proportions, by=...)``.
       **Different boundaries**: the item on which a threshold is crossed now belongs
       to the more popular group. Any number of groups is supported.
   * - ``ActivitySegmentation().segment(data, proportions)``
     - ``segmentation.by_interaction_count(data, proportions, by="user")``.
       **Different groups**: it is now deterministic (version 3 added random noise to
       the counts), and the boundaries are rounded up instead of to the nearest
       integer. The defaults changed too: ``proportions`` from ``[0.1, 0.9]`` to
       ``(0.2, 0.8)``, and the side from users to ``by="item"``.
   * - ``PopularityPercentage().segment(data, group=...)``
     - ``segmentation.popularity(data, by=...)``; the column is named ``popularity``
       instead of ``percentage``. The value is still the share of the interactions;
       ``relative_to="population"`` gives the fraction of users (or items) instead.
   * - ``DiscreteFeatureSegmentation().segment(feature, fill_na=-1)``
     - ``segmentation.by_feature(features, column)``. The attribute is the column
       named ``column`` instead of the second column of the table, and the
       identifiers are in a ``user`` or an ``item`` column. Groups keep the values of
       the attribute instead of integer codes; missing values form the group
       ``"unknown"``, or the value passed as ``missing`` (the former ``fill_na``).
   * - ``proportions`` with one value
     - Everything is in group 1; version 3 returned the interactions unchanged.
   * - ``min_interaction=...``
     - Removed: filter the interactions before segmenting.
   * - ``ColumnsNotExistException``
     - ``errors.MissingColumnsError``.
   * - ``SegmentationNotSupportedException``, ``WrongProportionsException``,
       ``InvalidValueException``, ``InvalidGroupException``
     - ``errors.InvalidInputError``.
   * - ``RecListTooShortException``
     - Removed: a cutoff beyond the length of the lists is accepted.
   * - ``AccuracyMetric``, ``BeyondAccuracyMetric``, ``FairnessMetric``,
       ``Segmentation``
     - Removed: there are no classes to derive from.

Both errors derive from ``errors.RecsyslearnError`` and from ``ValueError``.

Other changes
-------------

* Group tables are passed next to the recommendations instead of being merged into
  them, and group labels keep their type (segmentations return integers ``1, 2, ...``
  by default, no longer strings).
* A recommended user or item without a group is refused. In version 3 the merge with
  the group table dropped those recommendations silently.
* Identifiers keep their type: the segmentations of version 3 returned them as strings.
  Tables that are used together must hold identifiers of the same type; integers
  against strings are refused instead of being read as "no match".
* Input tables are checked: a missing identifier, a ``(user, item)`` pair listed twice
  in the recommendations or in the ground truth, and a ``rank``, ``score`` or
  ``rating`` column that does not hold numbers are errors, see :doc:`data`.
* Input tables are never modified. Version 3 overwrote the ``rank`` column of the
  recommendations in place.
* Every metric accepts a cutoff ``k``.
* Recommendations in which two items of a user have the same ``rank`` are refused.
* The explicit feedback of the interactions is in a ``rating`` column; ``rank`` now
  only means the position of a recommendation.
* Recommendations can be ordered by a ``score`` column instead of ``rank``. Scores can
  be tied, see :doc:`data`.
