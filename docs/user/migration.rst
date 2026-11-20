Migrating from version 3
========================

Version 4 is a rewrite: the metrics were re-derived from the literature, the core is
written in Rust, and the class-based API was replaced by plain functions. Results can
differ from version 3 where version 3 departed from the published definitions; those
cases are marked below.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Version 3
     - Version 4
   * - ``NDCG().evaluate(top_n, find_relevant_items(test), ats)``
     - ``accuracy.ndcg(top_n, test, k=...)``, one call per cutoff. It returns the mean;
       ``per_user=True`` returns the scores. If ``test`` has a ``rank`` column it is
       now used as graded relevance: drop the column to keep binary relevance.
   * - ``find_relevant_items``
     - Removed: pass the test interactions directly.
   * - ``Coverage().evaluate(top_n, items)``
     - ``beyond_accuracy.coverage(top_n, items, k=...)``.
   * - ``Novelty().evaluate(top_n_with_popularity)``
     - ``beyond_accuracy.novelty(top_n, train, k=...)``. **Different values**: the
       popularity is computed from the interactions as the fraction of users who
       interacted with the item, instead of being read from a column.
   * - ``Entropy().evaluate(top_n_with_groups)``
     - ``fairness.entropy(top_n, groups)``. **Different values**: version 3 weighted
       recommendations by their raw rank; version 4 uses the exposure
       ``1 / log2(1 + position)`` (or ``exposure="flat"``).
   * - ``KullbackLeibler().evaluate(top_n_with_groups, target_representation)``
     - ``fairness.kl_divergence(top_n, groups, target)``, where ``target`` maps each
       group to its probability and defaults to parity.
   * - ``MutualInformation().evaluate(top_n_with_groups, "user")``
     - ``fairness.mutual_information(top_n, user_groups=groups)``; with ``"item"``,
       ``item_groups=groups``. Both can now be given at once.
   * - ``rel_matrix=...``
     - ``ground_truth=test``.
   * - ``InteractionSegmentation().segment(data, proportions, group=...)``
     - ``segmentation.by_cumulative_interactions(data, proportions, by=...)``.
       **Different boundaries**: the item on which a threshold is crossed now belongs
       to the more popular group. Any number of groups is supported.
   * - ``ActivitySegmentation().segment(data, proportions)``
     - ``segmentation.by_interaction_count(data, proportions, by="user")``. It is now
       deterministic (version 3 added random noise to the counts).
   * - ``PopularityPercentage().segment(data, group=...)``
     - ``segmentation.popularity(data, by=...)``; the column is named ``popularity``.
   * - ``DiscreteFeatureSegmentation().segment(feature)``
     - ``segmentation.by_feature(features, column)``. Groups keep the values of the
       attribute instead of integer codes; missing values form the group ``"unknown"``.
   * - ``min_interaction=...``
     - Removed: filter the interactions before segmenting.
   * - ``ColumnsNotExistException``
     - ``errors.MissingColumnsError``.
   * - The other exceptions
     - ``errors.InvalidInputError``.

Other changes:

* Group tables are passed next to the recommendations instead of being merged into
  them, and group labels keep their type (segmentations return integers ``1, 2, ...``
  by default, no longer strings).
* Input tables are never modified. Version 3 overwrote the ``rank`` column of the
  recommendations in place.
* Every metric accepts a cutoff ``k``.
* Recommendations in which two items of a user have the same ``rank`` are refused.
