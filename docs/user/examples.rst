Examples
========

The scripts below are in the ``examples`` folder of the repository. Each one runs from
the root of the repository, e.g. ``python examples/02_accuracy.py``, and prints what it
computes; they are executed by the test-suite, so what is shown here is what runs.

They read the three files of ``examples/__assets__``: ``train_dataset.csv`` and
``test_dataset.csv`` are implicit feedback (columns ``user, item``), and ``top_k.csv``
holds the top-10 lists of a model (``user, item, rank``).

The input tables
----------------

On data small enough to be checked by hand: lists ordered by ``rank``, by ``score`` or by
the order of the rows, identifiers of any type, and graded relevance in ``rating``. The
format is described in :doc:`data`.

.. literalinclude:: ../../examples/01_input_tables.py
    :language: python

Accuracy
--------

:func:`~recsyslearn.accuracy.ndcg` at several cutoffs, a score per user and a bootstrap
interval from it, who is evaluated, and the linear and exponential gain on graded
relevance.

.. literalinclude:: ../../examples/02_accuracy.py
    :language: python

Beyond accuracy
---------------

:func:`~recsyslearn.beyond_accuracy.coverage` with different catalogs, and
:func:`~recsyslearn.beyond_accuracy.novelty`, also per user and for an item without
interactions.

.. literalinclude:: ../../examples/03_beyond_accuracy.py
    :language: python

Segmentations
-------------

The functions of :mod:`recsyslearn.segmentation`: groups by share of the interactions
and by share of the population, for items and for users, popularity as a number, and
groups from a categorical attribute.

.. literalinclude:: ../../examples/04_segmentations.py
    :language: python

Fairness
--------

The functions of :mod:`recsyslearn.fairness`: the distribution of exposure and of
effectiveness over groups of items and of users, its entropy, its divergence from
several targets, and the mutual information between groups of users and of items.

.. literalinclude:: ../../examples/05_fairness.py
    :language: python

Scores and ties
---------------

A popularity baseline given with a ``score`` column, in which many items are tied: the
expected value of the metrics, how to impose an order instead, and why coverage is
refused. The formulas are in :doc:`../methodology`.

.. literalinclude:: ../../examples/06_scores_and_ties.py
    :language: python

Comparing recommenders
----------------------

A model, a popularity baseline and a random one, on every metric, in one table.

.. literalinclude:: ../../examples/07_compare_recommenders.py
    :language: python

Errors
------

What is refused, and with which error. The complete list is in :ref:`input-errors`.

.. literalinclude:: ../../examples/08_errors.py
    :language: python
