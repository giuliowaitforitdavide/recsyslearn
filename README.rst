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


*recsyslearn* is a Python library designed to evaluate recommendation systems comprehensively.
It offers a set of tools to measure recommendation accuracy, coverage, novelty, and fairness.
This library is a valuable resource for data scientists and engineers who aim to enhance the performance
and fairness of their recommendation algorithms.


Key Features
------------


Dataset Utilities
^^^^^^^^^^^^^^^^^

*recsyslearn* simplifies the process of calculating item popularity and user activity, and of
segmenting (i.e., categorizing) users and items into groups. The users and items can be segmented
based on various criteria, hence providing the basis for group fairness analyses on
several dimensions.

In particular, the following type of segmentations are provided:
* item segmentation based on a popularity value, or user segmentation based on an activity value, corresponding to the percentage of user-item interactions.
* user or item segmentation based on one of their categorical features (e.g., user gender, or item genre).
* item segmentation based on the cumulative number of interactions of the items in each group. For instance, keeping the argument of the method to the default value of 80 − 20, the most popular items corresponding to the first group account for 80% of the interactions, and the items in the second group account for the remaining 20%.
* user segmentation based their grade of activity. For instance, keeping the argument of the method to the default value of 80 − 20, the most active 80% users will belong to the first group, and the least active 20% users to the second.


Accuracy Evaluation metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* NDCG@k: *recsyslearn* computes the Normalized Discounted Cumulative Gain (NDCG) metric to assess recommendation accuracy at a specific cutoff k. NDCG@k provides insights into the relevance of recommended items and their ranking.


Beyond Accuracy metrics
^^^^^^^^^^^^^^^^^^^^^^^

* Coverage: evaluate the coverage of your recommendation system using various metrics. These metrics measure the extent to which unique items are recommended to users and provide insights into the diversity of recommendations.
* Novelty: measure the novelty of recommendations to ensure that users receive fresh and engaging content.


*recsyslearn* helps you assess the diversity and freshness of recommended items.


Fairness metrics
^^^^^^^^^^^^^^^^

* Entropy: the measure of diversity (i.e., recommendations or accurate recommendations) over user or item groups.
* Mutual Information: measures to what extent the information on the user group provides information about the groups to which the recommendations belong.
* Kullback-Leibler: measures the KL divergence between the distribution of utility over user or item groups, computed on the list of recommendations, and a target distribution.


License
-------

Recsyslearn is released as free software under the GNU General Public License v3.

Documentation
-------------

For in-depth documentation, detailed explanations of functions, and usage examples, please visit the
`official documentation`_.

Citation
--------

If you use *recsyslearn* in your research, please cite the following paper:

.. code-block:: console
        
        @proceedings{Moscati2023MultiObjectiveHyperOpt,
        title = {Multiobjective Hyperparameter Optimization of Recommender Systems},
        author = {Moscati, Marta and Deldjoo, Yashar and Carparelli, Giulio Davide and Schedl, Markus},
        booktitle = {Proceedings of the 3rd Workshop on Perspectives on the Evaluation of Recommender Systems co-located with the 17th ACM Conference on Recommender Systems (RecSys 2023), Singapore, Singapore.},
        editor = {Said, Alan and Zangerle, Eva and Bauer, Christine},
        publisher = {CEUR-WS.org},
        url = {https://ceur-ws.org/Vol-3476/paper3.pdf},
        volume = {3476},
        year = {2023}
        }


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`official documentation`: https://recsyslearn.readthedocs.io/en/latest/?version=latest

