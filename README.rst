===========
Recsyslearn
===========


.. image:: https://github.com/giuliowaitforitdavide/recsyslearn/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/giuliowaitforitdavide/recsyslearn/actions/workflows/tests.yml
        :alt: Test Status

.. image:: https://readthedocs.org/projects/recsyslearn/badge/?version=latest
     :target: https://recsyslearn.readthedocs.io/en/latest/?version=latest
     :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/recsyslearn.svg
        :target: https://pypi.python.org/pypi/recsyslearn
        :alt: Library Version


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

Known Issues
------------

In this version of the library, the computation of the metrics for cross groups (user and item groups together) has not been implemented yet.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
