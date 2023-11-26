Welcome to recsyslearn's documentation!
=======================================

.. contents:: Table of Contents

.. toctree::
   :maxdepth: 1
   :hidden:
   
   API Reference <api_ref/index>
   User Guide  <user/index>

**Recsyslearn** is a Python library designed to evaluate and benchmark recommendation systems comprehensively.
It offers a set of tools to measure recommendation accuracy, coverage, novelty, and fairness.
This library is a valuable resource for data scientists and engineers who aim to enhance the performance
and fairness of their recommendation algorithms.

Key Features
============

Accuracy Evaluation metrics
----------------------------

- NDCG@N: "recsyslearn" computes the Normalized Discounted Cumulative Gain (NDCG) metric
to assess recommendation accuracy. NDCG@N provides insights into the relevance of recommended
items and their ranking.

Beyond Accuracy metrics
-----------------------

- Coverage: evaluate the coverage of your recommendation system using various metrics.
These metrics measure the extent to which unique items are recommended to users
and provide insights into the diversity of recommendations.
- Novelty: measure the novelty of recommendations to ensure that users receive fresh and engaging content. Recsyslearn helps you assess the diversity and freshness of recommended items.

Fairness metrics
----------------

- Entropy
- Mutual Information
- Kullback-Leibler Divergence

Fairness Grouping Utilities
---------------------------

Recsyslearn simplifies the process of dividing your dataset into groups. You can segment
data based on various criteria, such as user activity or item popularity, enabling in-depth
fairness analysis and adjustments.
