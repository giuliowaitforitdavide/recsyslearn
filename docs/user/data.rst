Input data
==========

Every function of recsyslearn takes pandas DataFrames in *long* format, one row per
``(user, item)`` pair, with fixed column names:

``user``
    The identifier of the user.
``item``
    The identifier of the item.

and, depending on the table, one optional column of numbers:

``rating``
    In interactions: the explicit feedback of the user on the item.
``rank`` or ``score``
    In recommendations: what orders the list of each user.

The names follow what most tools use: ``rating`` for explicit feedback and ``rank``
and ``score`` for the output of a recommender (LensKit, the TREC run format, ranx).

Identifiers can be of any hashable type (integers, strings, ...), as long as the same
type is used in all the tables passed to a function: two tables that have nothing in
common because one holds integers and the other strings are refused. The tables are
never modified, and columns that a function does not read are ignored, with two limits:
a column the function reads cannot appear twice, and a table of groups or of features
has *either* a ``user`` *or* an ``item`` column, since it describes one of the two
(see the functions that take them).

Identifiers must not be missing. The order of the rows does not change a result, with
the exceptions stated where they apply: recommendations with neither a ``rank`` nor a
``score`` column, whose row order *is* the ranking, and identifiers that cannot be compared with each
other (a number and a tuple in the same column), for which the last digits of a result
can depend on which comes first.

There are two kinds of tables.

Interactions
------------

What users did: the training set, the held-out test set, the catalog.

* **Implicit feedback** (clicks, plays, purchases): columns ``["user", "item"]``.
  Every row is an interaction.
* **Explicit feedback** (ratings): columns ``["user", "item", "rating"]``, where
  ``rating`` is the feedback value.

When interactions are used as the *ground truth* of a metric, the pairs of an implicit
table are the relevant ones, and the ``rating`` of an explicit table is the graded
relevance of the pair. Pairs that are not listed have relevance 0. To consider only
ratings above a threshold as relevant, filter the table before passing it:

.. code-block:: python

    relevant = test[test["rating"] >= 4].drop(columns="rating")

Where the feedback value is not used (the popularity of the items, the segmentations),
a ``rating`` column is ignored.

Recommendations
---------------

What the system suggested, one row per recommended item. The list of each user is
ordered by one of:

``rank``
    Columns ``["user", "item", "rank"]``: the best item has the smallest rank
    (1, 2, 3, ...).
``score``
    Columns ``["user", "item", "score"]``: the best item has the largest score, as
    returned by most models.
neither
    Columns ``["user", "item"]``: the order of the rows is the ranking, and the first
    row of a user is their best item.

When both ``rank`` and ``score`` are there, ``rank`` is used and ``score`` is not read:
like any column a function does not read, it is not checked either.

Only the order matters. Formulas use the *position* of an item in the sorted list
(1 for the first, 2 for the second, ...), never the value of ``rank`` or ``score``
itself: ranks that start from 0 or have gaps, and scores on any scale, give the same
results. A user cannot be recommended the same item twice. Integer ranks or scores
beyond :math:`2^{53}` are refused, because they are compared as floating-point numbers
and could not be told apart; infinite scores are allowed, missing ones are not.

Ties
^^^^

A ``rank`` is a position, so two items of a user cannot have the same one: the table is
refused instead of being ordered by the position of its rows.

Two items of a user *can* have the same ``score``; for a popularity baseline most of
them do. Their relative order is then unknown, and recsyslearn does not choose one:

* :func:`~recsyslearn.accuracy.ndcg` returns the value expected when every order of the
  tied items is equally likely, computed exactly and without sampling (McSherry &
  Najork, 2008). An item tied across the cutoff counts for the probability that it
  falls within it;
* :func:`~recsyslearn.beyond_accuracy.novelty` does the same. No paper treats ties for
  novelty: recsyslearn applies the formula of the nDCG to it, because novelty is a sum
  over positions too;
* the fairness metrics give each tied item the exposure it is expected to receive
  (Diaz et al., 2020) and are then computed on those expected exposures. The result is
  *not* the mean of the metric over the orders: two tied items of two groups with a
  cutoff of 1 give an entropy of 1 bit, while every single order gives 0;
* :func:`~recsyslearn.beyond_accuracy.coverage` has no published definition for that
  case, and refuses a table in which items with the same score straddle the cutoff.
  Ties that the cutoff does not split are no obstacle.

See :doc:`../methodology` for the formulas. To impose an order instead (for example to
reproduce a tool that breaks ties by identifier), compute the ranks yourself and pass
them as ``rank``.

Every metric has a cutoff ``k`` that restricts the evaluation to the first ``k``
positions of each list; by default the whole lists are used.

Groups
------

The fairness metrics also take a table that assigns users or items to groups: columns
``["user", "group"]`` or ``["item", "group"]``, one row per user (item). The functions
of :mod:`recsyslearn.segmentation` produce such tables, but any table of that shape
works.

.. _input-errors:

Errors
------

Every function checks its tables before computing anything, and raises

* ``TypeError`` when a table is not a pandas DataFrame;
* :class:`~recsyslearn.errors.MissingColumnsError` when a column the function needs is
  not there;
* :class:`~recsyslearn.errors.InvalidInputError` when

  * a column the function reads appears more than once;
  * an identifier is missing or is not hashable;
  * the identifiers of two tables are of different types and none of them match, or
    are numbers of different types (``int64`` and ``uint64``, integers and floats) with
    integers beyond :math:`2^{53}`, which would be merged once combined;
  * the columns have more than one level of names;
  * ``k`` is not a positive integer or ``None``;
  * a ``rank``, a ``score`` or a ``rating`` is missing, is in a column whose dtype is
    not an integer or floating-point one (strings of digits, booleans and dates are
    not converted), or holds integers beyond :math:`2^{53}`;
  * two recommendations of a user have the same ``rank``, or a ``(user, item)`` pair is
    listed twice in the recommendations or in the ground truth.

Both exceptions are also ``ValueError``. The errors specific to a function are listed in
its documentation. Messages name the argument and the column and give counts; they never
quote the content of the tables.

