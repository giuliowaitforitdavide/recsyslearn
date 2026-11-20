.. image:: https://raw.githubusercontent.com/giuliowaitforitdavide/recsyslearn/main/docs/_static/logo.svg
        :alt: recsyslearn logo
        :width: 96
        :align: center

.. The documentation includes this file from here: its pages already carry the logo.
.. after-logo

===========
Recsyslearn
===========


.. image:: https://github.com/giuliowaitforitdavide/recsyslearn/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/giuliowaitforitdavide/recsyslearn/actions/workflows/ci.yml
        :alt: CI Status

.. image:: https://readthedocs.org/projects/recsyslearn/badge/?version=stable
     :target: https://recsyslearn.readthedocs.io/en/stable/
     :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/recsyslearn.svg
        :target: https://pypi.python.org/pypi/recsyslearn
        :alt: Library Version


*recsyslearn* evaluates recommender systems: accuracy, coverage, novelty and group
fairness, plus the segmentations of users and items those analyses need.

* **Grounded in the literature.** Every metric is implemented from the formula of the
  paper that defines it, cited in its documentation, and tested against that formula.
  The choices the papers leave open (ties, undefined cases) are documented, not hidden.
* **Validated on published results.** Where a paper publishes its data, the numbers it
  prints are reproduced by the test-suite; the documentation says which results are
  reproduced and which cannot be.
* **Fast.** The computations run in a multi-threaded Rust core: on a laptop, a million
  recommendations are evaluated in a few tens of milliseconds.
* **Plain pandas in, plain numbers out.** Tables with the columns ``user`` and
  ``item``, an optional ``rating`` for explicit feedback, and ``rank`` or ``score`` to
  order the recommendations.


Installation
------------

.. code-block:: console

    $ pip install recsyslearn

Pre-built wheels are available for Linux, macOS and Windows on Python 3.11 and later.


Example
-------

.. code-block:: python

    import pandas as pd
    from recsyslearn import accuracy, beyond_accuracy, fairness, segmentation

    # Training and held-out interactions: user, item[, rating].
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    # Recommendations: user, item, and rank (1 = best) or score (highest = best).
    top_k = pd.read_csv("top_k.csv")

    accuracy.ndcg(top_k, test, k=10)
    beyond_accuracy.coverage(top_k, train, k=10)
    beyond_accuracy.novelty(top_k, train, k=10)

    # Is exposure spread fairly between popular and niche items?
    item_groups = segmentation.by_cumulative_interactions(train, (0.8, 0.2))
    fairness.kl_divergence(top_k, item_groups)


Features
--------

All the tables are pandas DataFrames with the columns ``["user", "item"]`` for implicit
feedback or ``["user", "item", "rating"]`` for explicit feedback; recommendations are
ordered by a ``rank`` or by a ``score`` column, and scores can be tied.

Accuracy
^^^^^^^^

* **nDCG@k**, with linear or exponential gain and binary or graded relevance.

Beyond accuracy
^^^^^^^^^^^^^^^

* **Coverage**: the share of the catalog that is recommended to at least one user.
* **Novelty**: the mean self-information of the recommended items, which is high when
  the system recommends items that few users know.

Fairness
^^^^^^^^

The fairness metrics look at how the benefit of being recommended (the exposure in the
lists, or the exposure of *relevant* recommendations) is distributed over groups of
users or groups of items.

* **Entropy**: how evenly the benefit is spread over the groups.
* **Kullback-Leibler divergence**: how far the distribution is from a target, such as
  parity or proportionality to the size of the groups.
* **Mutual information**: how much the group of a user determines the groups of the
  items they are recommended.

Segmentations
^^^^^^^^^^^^^

* **By cumulative interactions**: e.g. the short head of items that account for 80% of
  the interactions and the long tail, or head / mid / tail.
* **By number of interactions**: e.g. the 20% most popular items, or the 5% most active
  users.
* **Popularity**: the share of interactions, or of users, of each item.
* **By a discrete feature**: e.g. the gender of the users or the genre of the items.


Documentation
-------------

The `official documentation`_ contains the user guide, the formula and the source of
every metric, the published results the library is validated against, and the guide to
migrate from version 3.


Contributing
------------

Contributions are welcome: see `CONTRIBUTING.md`_. The development guidelines (workflow,
architecture, testing and security) are in `docs/development`_, and `AGENTS.md`_ is the
entry point for coding agents. Security problems are reported as described in
`SECURITY.md`_.


License
-------

Recsyslearn is released as free software under the MIT License.


Citation
--------

If you use *recsyslearn* in your research, please cite the following paper:

.. code-block:: bibtex

        @inproceedings{Moscati2023MultiObjectiveHyperOpt,
        title = {Multiobjective Hyperparameter Optimization of Recommender Systems},
        author = {Moscati, Marta and Deldjoo, Yashar and Carparelli, Giulio Davide and Schedl, Markus},
        booktitle = {Proceedings of the 3rd Workshop on Perspectives on the Evaluation of Recommender Systems co-located with the 17th ACM Conference on Recommender Systems (RecSys 2023), Singapore, Singapore.},
        editor = {Said, Alan and Zangerle, Eva and Bauer, Christine},
        publisher = {CEUR-WS.org},
        url = {https://ceur-ws.org/Vol-3476/paper3.pdf},
        volume = {3476},
        year = {2023}
        }


.. _`official documentation`: https://recsyslearn.readthedocs.io/en/stable/
.. _`CONTRIBUTING.md`: https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/CONTRIBUTING.md
.. _`docs/development`: https://github.com/giuliowaitforitdavide/recsyslearn/tree/main/docs/development
.. _`AGENTS.md`: https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/AGENTS.md
.. _`SECURITY.md`: https://github.com/giuliowaitforitdavide/recsyslearn/blob/main/SECURITY.md
