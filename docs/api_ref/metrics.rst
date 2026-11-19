Metrics
=======

Every metric returns a ``MetricResult``. Per-user metrics (``NDCG``, ``Novelty``)
fill ``per_user`` and set ``aggregate`` to the mean over users; population-level
metrics (``Coverage``, ``Entropy``, ``KullbackLeibler``, ``MutualInformation``)
set ``per_user`` to ``None``.

.. automodule:: recsyslearn.result
   :members: MetricResult

.. automodule:: recsyslearn.accuracy.metrics
   :members: NDCG
   :show-inheritance:

.. automodule:: recsyslearn.beyond_accuracy.metrics
   :members: Coverage, Novelty
   :show-inheritance:

.. automodule:: recsyslearn.fairness.metrics
   :members: Entropy, KullbackLeibler, MutualInformation
   :show-inheritance: