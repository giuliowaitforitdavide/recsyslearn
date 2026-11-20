Methodology
===========

recsyslearn is developed with a fixed procedure, so that every number it returns can be
traced to a published definition:

1. the formula of a metric, or the procedure of a segmentation, is taken from the paper
   that introduced it (or from the paper that states it most precisely);
2. unit tests are written from the formula: values worked out by hand on small examples,
   and a comparison against a naive, loop-based transcription of the formula on random
   data;
3. only then the function is implemented in the Rust core;
4. where a paper publishes the data behind the numbers it prints, a test reproduces those
   numbers from those data (see `Validation against published results`_).

This page lists, for each function, the definition, its source, and the choices that
had to be made where the sources are silent.

Notation: :math:`U` and :math:`I` are the users and the items, :math:`L_u` is the
recommendation list of user :math:`u` cut at :math:`k`, and :math:`pos_u(i)` the
position (1 = best) of item :math:`i` in it. Logarithms are in base 2.

Accuracy
--------

:func:`recsyslearn.accuracy.ndcg`

.. math::

    \mathrm{DCG@}k(u) = \sum_{p=1}^{\min(k, |L_u|)} \frac{g(rel_u(L_u[p]))}{\log_2(p + 1)}
    \qquad
    \mathrm{nDCG@}k(u) = \frac{\mathrm{DCG@}k(u)}{\mathrm{IDCG@}k(u)}

* Cumulated gain normalised by the ideal ranking: [Jarvelin2002]_. The ideal ranking
  is built from all the relevant items of the user (their "recall base"), not from the
  recommended ones.
* Discount :math:`1 / \log_2(p + 1)`: [Burges2005]_. [Wang2013]_ calls the logarithmic
  discount :math:`1 / \log(p + 1)` the standard one; the base of the logarithm cancels
  in the ratio, so it does not change the metric. The original formulation of [Jarvelin2002]_ divides by :math:`\log_b p`
  and does not discount the positions before :math:`b`; it is not implemented.
* Gain: linear, :math:`g(r) = r` [Jarvelin2002]_, or exponential, :math:`g(r) = 2^r - 1`
  [Burges2005]_.
* Implicit feedback: :math:`\mathrm{IDCG@}k(u) = \sum_{p=1}^{\min(k, |Rel_u|)} 1 / \log_2(p + 1)`,
  Eq. 8 of [Krichene2020]_.

Choices: users without relevant items are left out of the mean, since the metric is
0/0 for them (the convention of ``trec_eval``); users with relevant items but no
recommendations score 0; positions missing from a list shorter than :math:`k` add no
gain; a relevance whose exponential gain is not a finite number (:math:`r \geq 1024`)
is refused, and so are gains so large that their sum is not one (the score would be
:math:`\infty / \infty`).

Order of the lists and tied scores
----------------------------------

The list of a user is ordered by ascending ``rank`` or by descending ``score``. A rank
is a position: two recommendations of a user with the same ``rank`` are refused. Scores
can be equal, and the relative order of the tied items is then unknown. ``trec_eval``
sorts by score and breaks ties by document identifier; that makes the result depend on
how the items happen to be named, so recsyslearn does not choose an order at all.

Following [McSherry2008]_, the value of a metric on a list with ties is its mean over
all the orders of the tied items, each equally likely, obtained analytically. With the
list split into groups :math:`V_1, V_2, \dots` of tied items, :math:`V_i` spanning the
positions :math:`t_i + 1, \dots, t_{i+1}`, Section 2.6 of the paper gives, for a gain
:math:`G` and a discount :math:`D`,

.. math::

    \mathrm{DCG@}k = \sum_i \left[ \frac{1}{|V_i|} \sum_{v \in V_i} G(v) \right]
    \left[ \sum_{j = t_i + 1}^{\min(t_{i+1}, k)} D(j) \right]

i.e. every item of a tie receives the mean discount of the positions the tie spans,
counting as zero those after the cutoff. Without ties it is the usual sum.

* :func:`recsyslearn.accuracy.ndcg`: the formula above; the ideal DCG is unchanged
  ("the normalization step is independent of ties", [McSherry2008]_). For one relevant
  item it reduces to Eq. 2 of [Guo2026]_.
* Fairness: the exposure of an item becomes its *expected* exposure over the orders,
  which is how [Diaz2020]_ (Eq. 4) define the exposure of an item under a distribution
  over rankings; here the distribution is uniform over the orders of the ties and the
  exposure of a position is :math:`\phi`. The benefit is that expected exposure, times
  the relevance for effectiveness.
* :func:`recsyslearn.beyond_accuracy.novelty`: no source treats ties for novelty. It is
  a sum over positions like the DCG, with the self-information as gain and
  :math:`1 / |L_u|` as the weight of every position within the cutoff, so the formula
  above applies as it is: an item tied across the cutoff counts for the probability
  that it is within it.
* :func:`recsyslearn.beyond_accuracy.coverage`: no source treats ties, and the number of
  distinct items is not a sum over positions, so the formula does not apply. Lists in
  which items with the same score straddle the cutoff are refused; ties that the cutoff
  does not split do not change the set of recommended items.

Choices: every order of a tie is equally likely, independently for each user; the
probability of a group in the fairness metrics is the ratio of the expected benefits,
not the expectation of the ratio (the two coincide for the exposure, whose total does
not depend on the order); ``rank`` is used when both columns are given; a missing score
is refused, an infinite one is ordered like any other.

Beyond accuracy
---------------

:func:`recsyslearn.beyond_accuracy.coverage`

.. math::

    \mathrm{coverage@}k = \frac{\left| \bigcup_{u \in U} L_u \right|}{|I|}

Catalog coverage of [Ge2010]_: :math:`I` is the set of all the available items.

Choices: the catalog is passed explicitly, because the items that could be recommended
cannot be deduced from the recommendations; an identifier repeated in it counts once;
a recommended item that is not in the catalog is refused, so that the result never
exceeds 1.

:func:`recsyslearn.beyond_accuracy.novelty`

.. math::

    \mathrm{novelty@}k = \frac{1}{|U_L|} \sum_{u \in U_L} \frac{1}{|L_u|} \sum_{i \in L_u} \log_2 \frac{|U|}{|U_i|}

The mean self-information of the recommended items [Zhou2010]_, where :math:`U_i` are
the users who interacted with item :math:`i`; the same formula is Eq. 12 of
[Schedl2018]_, with popularity defined as the fraction of users who rated the item, and
it is the "inverse user frequency" of [Vargas2011]_.

Choices: the self-information of an item nobody interacted with diverges and no source
says how to treat it; it is set to :math:`\log_2 |U|`, the value of an item with one
user. (RankSys and Elliot similarly fall back to the largest observed novelty.) The
mean is over the users who received recommendations.

Fairness
--------

The metrics implement the framework of [Amigo2023]_, which is the reference used by
[Moscati2023]_. A recommended pair carries a benefit :math:`\theta(u, i)`:

* exposure, :math:`\phi(u, i) = 1 / \log_2(1 + pos_u(i))`, the rank discount used by
  [Singh2018]_ and by the reference implementation of [Amigo2023]_, or the flat
  :math:`\phi(u, i) = 1` within the cutoff;
* effectiveness, :math:`\phi(u, i) \cdot \psi(u, i)`, where the utility :math:`\psi` is
  the relevance of the pair in the ground truth.

:func:`recsyslearn.fairness.group_distribution`

.. math::

    P(u, i) = \frac{\theta(u, i)}{\sum_{u', i'} \theta(u', i')}
    \qquad
    P(g) = \sum_{(u, i) \in g} P(u, i)

The benefit is normalised over all the recommended pairs, and the probability of a
group of users (or of items) is the sum over the pairs of its members [Amigo2023]_.
The three metrics below describe this distribution.

:func:`recsyslearn.fairness.kl_divergence` — inequity, Eq. 1 of [Amigo2023]_:

.. math::

    D_{KL}(P \,\|\, Q) = \sum_{g} P(g) \log_2 \frac{P(g)}{Q(g)}

It is the divergence of the observed distribution :math:`P` from the target :math:`Q`,
as in [Amigo2023]_ and [Moscati2023]_; [Steck2018]_ uses the opposite direction.

:func:`recsyslearn.fairness.mutual_information` — dependence, Eq. 2 of [Amigo2023]_:

.. math::

    I(X; Y) = \sum_{x, y} P(x, y) \log_2 \frac{P(x, y)}{P(x) P(y)}

between a partition of the users and a partition of the items; either can be the
partition into individuals.

:func:`recsyslearn.fairness.entropy`

.. math::

    H = -\sum_{g} P(g) \log_2 P(g)

[Amigo2023]_ gives no formula for an entropy-based metric; this is the entropy of
[Shannon1948]_ applied to the same distribution :math:`P`, as described in the appendix
of [Moscati2023]_. The normalised variant divides by :math:`\log_2 n` as in [Patro2020]_.

Choices: :math:`0 \log 0 = 0`; the divergence is infinite when the target excludes a
group that receives benefit, and no smoothing is applied; the base of the logarithm,
which [Amigo2023]_ leaves unspecified, is 2 as in its reference implementation; without
a target the divergence is measured from parity, :math:`Q(g) = 1/n`; a target must sum
to 1 within :math:`10^{-6}`, which admits shares rounded to six decimals.

Segmentation
------------

:func:`recsyslearn.segmentation.by_cumulative_interactions`

Items are sorted by decreasing number of interactions and cut where the cumulative
share of interactions reaches the given proportions. Short head = 80% of the ratings:
[Abdollahpouri2019a]_. Head / mid / tail = 20% / 60% / 20% of the ratings:
[Abdollahpouri2021]_. Cutting the popularity curve on cumulative volume goes back to
[Celma2008]_.

Choices: the item on which a threshold is crossed belongs to the more popular group
(the head is the smallest set of items that *reaches* its share, consistent with the
examples of [Abdollahpouri2021]_ and with the :math:`N_{50}` of [Celma2008]_).

:func:`recsyslearn.segmentation.by_interaction_count`

Items or users are sorted by decreasing number of interactions and cut at the given
shares of the population: the 5% most active users of [Li2021]_, the 20% most popular
items of [Abdollahpouri2019b]_.

Choices: the *cumulative* boundaries are rounded up, so that the first :math:`g` groups
together hold the top :math:`\lceil (p_1 + \dots + p_g) \cdot n \rceil` entities and
the first group is never empty. The size of a single group is the difference of two
boundaries, not :math:`\lceil p_g \cdot n \rceil`: with proportions (0.3, 0.3, 0.4) and
5 items the boundaries are 2 and 3 and the sizes 2, 1 and 2; with few entities a group
after the first can be empty.

In both segmentations, entities with the same number of interactions are ordered by
ascending identifier. No source specifies the order of ties; this one makes the result
independent of the order of the rows.

Proportions are floating-point numbers, so two tolerances are applied in both
segmentations: they must sum to 1 within :math:`10^{-9}`, and a threshold that falls
within rounding error of an integer (a few units in the last place for every proportion
summed) is taken to be that integer. Without the second one, ``0.1 + 0.2`` of 10 items
would be 3.0000000000000004 and round up to 4.

:func:`recsyslearn.segmentation.popularity`

The share of interactions, :math:`n_i / N` [Vargas2011]_, or the fraction of users,
:math:`|U_i| / |U|` [Zhou2010]_ [Abdollahpouri2019b]_.

:func:`recsyslearn.segmentation.by_feature`

Groups are the values of a categorical attribute, e.g. gender or age bracket
[Ekstrand2018]_, or item category [Deldjoo2021]_. Missing values form their own group,
as in [Ekstrand2018]_.

Validation against published results
------------------------------------

Besides the unit tests, the integration tests in ``tests/integration`` start from the
data published with a paper and check that recsyslearn returns the numbers printed in
it. They download the data on first use and are run with ``pytest -m integration``.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Functions
     - Published result
     - Outcome
   * - :func:`~recsyslearn.fairness.kl_divergence`,
       :func:`~recsyslearn.fairness.mutual_information`,
       :func:`~recsyslearn.fairness.group_distribution`
     - Tables 5 and 6 of [Amigo2023]_: 15 fairness measures of 8 recommenders on
       CiteULike and Netflix, from the recommendation lists, test ratings and groups
       published by the authors.
     - All the 240 printed values are reproduced, as are the 360 values of the output of
       the authors' implementation stored in their repository. The published ranks
       passed as scores give identical results.
   * - :func:`~recsyslearn.fairness.group_distribution`
     - The toy example of [Deldjoo2019]_ (Tables 1 and 2).
     - The 9 printed values of the generalized cross entropy follow from the group
       distributions computed by recsyslearn.
   * - :func:`~recsyslearn.beyond_accuracy.coverage`
     - Item coverage of the most-popular baseline on MovieLens 1M and Epinions, Tables 9
       and 11 of [Anelli2022]_, from the 5 published folds.
     - The 4 printed values are reproduced. One (MovieLens 1M at 20) depends on the
       order of equally popular items. With the popularity passed as ``score``, the two
       values at 20 are reproduced without assuming an order, and at 10 the input is
       refused because ties straddle the cutoff.
   * - :func:`~recsyslearn.accuracy.ndcg` with tied scores
     - The toy example of Section 3.2 of [Guo2026]_: one relevant item among 31 tied
       candidates, expected NDCG@10 = 0.1466.
     - The printed value is reproduced, and so is the closed form of the paper's Eq. 2
       for other sizes of the tie.
   * - :func:`~recsyslearn.accuracy.ndcg`
     - nDCG@1, 5 and 10 of the most-popular baseline on MovieLens 1M and Pinterest,
       Tables 8 and 9 of [FerrariDacrema2021]_, from the data published with [He2017]_.
     - The 6 printed values lie between the results of the two extreme orders of equally
       popular candidates, which differ by less than 0.001. The fourth decimal depends
       on that order, which the papers do not document. With the popularity passed as
       ``score``, the expected nDCG over the orders lies between the same two extremes.

No published result could be reproduced for the following functions.

* :func:`~recsyslearn.beyond_accuracy.novelty`: [Zhou2010]_ prints values of exactly this
  metric, but the subsets of the data and the random split it used are not public; the
  other sources found either use a different definition (weighted by relevance and
  rank, or normalised differently) or print no value.
* :func:`~recsyslearn.segmentation.by_cumulative_interactions`: the thresholds printed in
  [Abdollahpouri2019a]_ and [Abdollahpouri2021]_ for MovieLens 1M do not follow from the
  public dataset under any boundary convention (the items with more than 506 ratings
  account for 59% of the ratings, not 80%; the 111 most rated items for 19.8%, not more
  than 20%).
* :func:`~recsyslearn.segmentation.by_interaction_count`: the papers found either print
  no group sizes, or do not publish the training set the groups are computed on, or rank
  users by the sum of their ratings rather than by their number of interactions.
* Novelty and the fairness metrics *with tied scores*: no paper found prints the value
  expected over the orders of the ties for them. They are checked against the mean
  over every order, enumerated, on small random data.
* :func:`~recsyslearn.fairness.entropy`: no paper with public data prints it. It is
  covered indirectly: with a uniform target, the divergence reproduced above equals
  :math:`\log_2 n - H`.

References
----------

.. [Anelli2022] V. W. Anelli, A. Bellogín, T. Di Noia, D. Jannach, C. Pomo. *Top-N
   recommendation algorithms: a quest for the state-of-the-art*. UMAP 2022.
   https://arxiv.org/abs/2203.01155
.. [Deldjoo2019] Y. Deldjoo, V. W. Anelli, H. Zamani, A. Bellogín, T. Di Noia.
   *Recommender systems fairness evaluation via generalized cross entropy*. RMSE
   workshop @ RecSys 2019. https://arxiv.org/abs/1908.06708
.. [Diaz2020] F. Diaz, B. Mitra, M. D. Ekstrand, A. J. Biega, B. Carterette. *Evaluating
   stochastic rankings with expected exposure*. CIKM 2020.
   https://arxiv.org/abs/2004.13157
.. [FerrariDacrema2021] M. Ferrari Dacrema, S. Boglio, P. Cremonesi, D. Jannach. *A
   troubling analysis of reproducibility and progress in recommender systems research*.
   ACM Transactions on Information Systems 39(2), 2021. https://arxiv.org/abs/1911.07698
.. [Guo2026] C. Guo, H. Chen, Y. Zhu, Y. Li. *Tie handling is part of the evaluation
   protocol: an order-invariance audit for tie-heavy recommender scores*. FRAME workshop
   @ RecSys 2026. https://arxiv.org/abs/2609.26977
.. [He2017] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, T.-S. Chua. *Neural collaborative
   filtering*. WWW 2017. https://arxiv.org/abs/1708.05031

.. [Abdollahpouri2019a] H. Abdollahpouri, R. Burke, B. Mobasher. *Managing popularity
   bias in recommender systems with personalized re-ranking*. FLAIRS 2019.
   https://arxiv.org/abs/1901.07555
.. [Abdollahpouri2019b] H. Abdollahpouri, M. Mansoury, R. Burke, B. Mobasher. *The
   unfairness of popularity bias in recommendation*. RMSE workshop @ RecSys 2019.
   https://arxiv.org/abs/1907.13286
.. [Abdollahpouri2021] H. Abdollahpouri, M. Mansoury, R. Burke, B. Mobasher, E.
   Malthouse. *User-centered evaluation of popularity bias in recommender systems*.
   UMAP 2021. https://arxiv.org/abs/2103.06364
.. [Amigo2023] E. Amigó, Y. Deldjoo, S. Mizzaro, A. Bellogín. *A unifying and general
   account of fairness measurement in recommender systems*. Information Processing &
   Management 60(1), 2023. https://doi.org/10.1016/j.ipm.2022.103115
.. [Burges2005] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, G.
   Hullender. *Learning to rank using gradient descent*. ICML 2005.
   https://doi.org/10.1145/1102351.1102363
.. [Celma2008] Ò. Celma, P. Cano. *From hits to niches? Or how popular artists can bias
   music recommendation and discovery*. Netflix Prize workshop @ KDD 2008.
   https://mtg.upf.edu/files/publications/Celma-ACM-Netflix-KDD2008.pdf
.. [Deldjoo2021] Y. Deldjoo, V. W. Anelli, H. Zamani, A. Bellogín, T. Di Noia. *A
   flexible framework for evaluating user and item fairness in recommender systems*.
   User Modeling and User-Adapted Interaction 31, 2021.
   https://doi.org/10.1007/s11257-020-09285-1
.. [Ekstrand2018] M. D. Ekstrand et al. *All the cool kids, how do they fit in?:
   cold-start recommendations, online user-level, and demographic effects on recommender
   evaluation*. FAT* 2018. https://proceedings.mlr.press/v81/ekstrand18b.html
.. [Ge2010] M. Ge, C. Delgado-Battenfeld, D. Jannach. *Beyond accuracy: evaluating
   recommender systems by coverage and serendipity*. RecSys 2010.
   https://doi.org/10.1145/1864708.1864761
.. [Jarvelin2002] K. Järvelin, J. Kekäläinen. *Cumulated gain-based evaluation of IR
   techniques*. ACM Transactions on Information Systems 20(4), 2002.
   https://doi.org/10.1145/582415.582418
.. [Krichene2020] W. Krichene, S. Rendle. *On sampled metrics for item recommendation*.
   KDD 2020. https://doi.org/10.1145/3394486.3403226
.. [Li2021] Y. Li, H. Chen, Z. Fu, Y. Ge, Y. Zhang. *User-oriented fairness in
   recommendation*. WWW 2021. https://arxiv.org/abs/2104.10671
.. [McSherry2008] F. McSherry, M. Najork. *Computing information retrieval performance
   measures efficiently in the presence of tied scores*. ECIR 2008.
   https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ecir2008.pdf
.. [Moscati2023] M. Moscati, Y. Deldjoo, G. D. Carparelli, M. Schedl. *Multiobjective
   hyperparameter optimization of recommender systems*. PERSPECTIVES workshop @ RecSys
   2023. https://ceur-ws.org/Vol-3476/paper3.pdf
.. [Patro2020] G. K. Patro, A. Biswas, N. Ganguly, K. P. Gummadi, A. Chakraborty.
   *FairRec: two-sided fairness for personalized recommendations in two-sided
   platforms*. WWW 2020. https://arxiv.org/abs/2002.10764
.. [Schedl2018] M. Schedl, H. Zamani, C.-W. Chen, Y. Deldjoo, M. Elahi. *Current
   challenges and visions in music recommender systems research*. International Journal
   of Multimedia Information Retrieval 7, 2018.
   https://doi.org/10.1007/s13735-018-0154-2
.. [Shannon1948] C. E. Shannon. *A mathematical theory of communication*. Bell System
   Technical Journal 27, 1948.
.. [Singh2018] A. Singh, T. Joachims. *Fairness of exposure in rankings*. KDD 2018.
   https://arxiv.org/abs/1802.07281
.. [Steck2018] H. Steck. *Calibrated recommendations*. RecSys 2018.
.. [Vargas2011] S. Vargas, P. Castells. *Rank and relevance in novelty and diversity
   metrics for recommender systems*. RecSys 2011.
   https://doi.org/10.1145/2043932.2043955
.. [Wang2013] Y. Wang, L. Wang, Y. Li, D. He, W. Chen, T.-Y. Liu. *A theoretical
   analysis of NDCG type ranking measures*. COLT 2013. https://arxiv.org/abs/1304.6480
.. [Zhou2010] T. Zhou, Z. Kuscsik, J.-G. Liu, M. Medo, J. R. Wakeling, Y.-C. Zhang.
   *Solving the apparent diversity-accuracy dilemma of recommender systems*. PNAS
   107(10), 2010. https://doi.org/10.1073/pnas.1000488107
