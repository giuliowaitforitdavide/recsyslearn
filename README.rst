===========
recsyslearn
===========


.. image:: https://img.shields.io/pypi/v/recsyslearn.svg
        :target: https://pypi.python.org/pypi/recsyslearn

.. image:: https://img.shields.io/travis/giuliowaitforitdavide/recsyslearn.svg
        :target: https://travis-ci.com/giuliowaitforitdavide/recsyslearn

.. image:: https://readthedocs.org/projects/recsyslearn/badge/?version=latest
        :target: https://recsyslearn.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




A small library to compute fairness of recommender systems.


* Free software: GNU General Public License v3
* Documentation: https://recsyslearn.readthedocs.io.


Features
--------

* Compute Novelty of a recommender system based on its recommendations list.
* Compute Coverage of a recommender system based on its recommendations list.
* Compute Entropy of a recommender system based on its recommendations list.
* Compute Kullback-Leibler divergence of a recommender system based on its recommendations list and the wanted target representation.
* Compute Mutual Information of a recommender system based on its recommendations list.
* Segment an implicit or explicit dataset in groups based on the activity of the users or on the popularity of the items.


Installation
------------

To install the library simply run in the command-line::

    pip install recsyslearn

Usage
-----

You just need a recommendation list in the form of a ``user, item, rank, group`` Dataframe. The library will do the rest.
If you don't have the info about the groups, you can use the library itself to segment the dataset. The dataset has to be in the form of ``user, item, rank``.

Known Issues
------------

In this version of the library, the computation of the metrics for cross groups (user and item groups together) has not yet implemented.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
