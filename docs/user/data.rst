Input data
==========

Every function of recsyslearn takes pandas DataFrames in *long* format, one row per
``(user, item)`` pair, with fixed column names:

``user``
    The identifier of the user.
``item``
    The identifier of the item.
``rank`` (optional)
    A number attached to the pair. What it means depends on the table, see below.

Identifiers can be of any hashable type (integers, strings, ...), as long as the same
type is used in all the tables passed to a function: two tables that have nothing in
common because one holds integers and the other strings are refused. The tables are
never modified, and columns that a function does not read are ignored, with two limits:
a column the function reads cannot appear twice, and a table of groups or of features
has *either* a ``user`` *or* an ``item`` column, since it describes one of the two
(see the functions that take them).

Identifiers must not be missing. The order of the rows does not change a result, with
the exceptions stated where they apply: recommendations without a ``rank`` column,
whose row order *is* the ranking, and identifiers that cannot be compared with each
other (a number and a tuple in the same column), for which the last digits of a result
can depend on which comes first.

There are two kinds of tables.

Interactions
------------

What users did: the training set, the held-out test set, the catalog.

* **Implicit feedback** (clicks, plays, purchases): columns ``["user", "item"]``.
  Every row is an interaction.
* **Explicit feedback** (ratings): columns ``["user", "item", "rank"]``, where ``rank``
  is the feedback value, e.g. the rating.

When interactions are used as the *ground truth* of a metric, the pairs of an implicit
table are the relevant ones, and the ``rank`` of an explicit table is the graded
relevance of the pair. Pairs that are not listed have relevance 0. To consider only
ratings above a threshold as relevant, filter the table before passing it:

.. code-block:: python

    relevant = test[test["rank"] >= 4].drop(columns="rank")

Recommendations
---------------

What the system suggested: columns ``["user", "item", "rank"]``, where ``rank`` orders
the list of each user and the best item has the smallest rank (1, 2, 3, ...).

Without the ``rank`` column, the order of the rows is the ranking: the first row of a
user is their best item.

Only the order matters. Formulas use the *position* of an item in the sorted list
(1 for the first, 2 for the second, ...), never the value of ``rank`` itself, so ranks
that start from 0 or have gaps give the same results. A user cannot be recommended the
same item twice, and two items of a user cannot have the same ``rank``: their order
would be undefined, so the table is refused instead of being ordered by the position of
its rows. Integer ranks beyond :math:`2^{53}` are refused as well, because they are
compared as floating-point numbers and could not be told apart.

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
  * the identifiers of two tables are of different types and none of them match;
  * ``k`` is not a positive integer or ``None``;
  * a ``rank`` is not a number or is missing, or the ranks are integers beyond
    :math:`2^{53}`;
  * two recommendations of a user have the same ``rank``, or a ``(user, item)`` pair is
    listed twice in the recommendations or in the ground truth.

Both exceptions are also ``ValueError``. The errors specific to a function are listed in
its documentation. Messages name the argument and the column and give counts; they never
quote the content of the tables.

