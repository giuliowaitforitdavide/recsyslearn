"""Accuracy metrics: how well the recommendations rank the relevant items."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from recsyslearn import _core
from recsyslearn._frames import Recommendations, check_choice, check_k
from recsyslearn.errors import InvalidInputError

__all__ = ["ndcg"]


def ndcg(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int | None = None,
    *,
    gain: Literal["linear", "exponential"] = "linear",
    per_user: bool = False,
) -> float | pd.Series:
    r"""Normalised discounted cumulative gain at ``k``.

    For a user :math:`u` with recommendation list :math:`L_u` and relevance
    :math:`rel_u(i)` in the ground truth,

    .. math::

        \mathrm{DCG@}k(u) = \sum_{p=1}^{\min(k, |L_u|)}
            \frac{g(rel_u(L_u[p]))}{\log_2(p + 1)},
        \qquad
        \mathrm{nDCG@}k(u) = \frac{\mathrm{DCG@}k(u)}{\mathrm{IDCG@}k(u)}

    where :math:`\mathrm{IDCG@}k(u)` is the DCG@k of the ideal ranking: *all* the
    ground-truth items of the user sorted by decreasing gain. With implicit feedback
    this is :math:`\sum_{p=1}^{\min(k, |Rel_u|)} 1 / \log_2(p + 1)`.

    Parameters
    ----------
    recommendations
        Columns ``["user", "item", "rank"]``: lists are ordered by ascending ``rank``,
        and two items of a user cannot have the same rank. Without ``rank``, the order
        of the rows is the ranking.
    ground_truth
        Held-out interactions. Columns ``["user", "item"]`` for implicit feedback
        (every pair is relevant) or ``["user", "item", "rank"]`` for explicit
        feedback, where ``rank`` is the graded relevance (e.g. the rating) of the pair.
    k
        Cutoff; ``None`` evaluates the whole lists.
    gain
        ``"linear"`` uses :math:`g(r) = r` (Järvelin & Kekäläinen, 2002),
        ``"exponential"`` uses :math:`g(r) = 2^r - 1` (Burges et al., 2005). They
        coincide for implicit feedback.
    per_user
        Return the score of every user instead of their mean.

    Returns
    -------
    float or pandas.Series
        The mean nDCG@k over the users with at least one relevant item in the ground
        truth, or their scores indexed by user. Users without relevant items are left
        out (the metric is undefined for them); users with relevant items but no
        recommendations score 0.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        if ``gain`` is not one of the two options; if a relevance is negative, not
        finite, or too large for the exponential gain (:math:`2^r - 1` must be a finite
        number, i.e. :math:`r < 1024`); if the gains of a user are too large to be added
        up; or if no user has a relevant item.

    References
    ----------
    * K. Järvelin and J. Kekäläinen. *Cumulated gain-based evaluation of IR
      techniques*. ACM TOIS 20(4), 2002. https://doi.org/10.1145/582415.582418
    * C. Burges et al. *Learning to rank using gradient descent*. ICML 2005.
      https://doi.org/10.1145/1102351.1102363
    * W. Krichene and S. Rendle. *On sampled metrics for item recommendation*
      (Eq. 8). KDD 2020. https://doi.org/10.1145/3394486.3403226
    """
    k = check_k(k)
    check_choice("gain", gain, ("linear", "exponential"))
    data = Recommendations(recommendations, ground_truth)

    scores = _core.ndcg(data.lists, data.truth, k, gain)
    mean = _core.mean_over_users(scores)
    if mean is None:
        raise InvalidInputError("ground_truth contains no relevant item: nDCG is undefined")
    if per_user:
        evaluated = ~np.isnan(scores)
        return pd.Series(scores[evaluated], index=data.users[evaluated], name="ndcg")
    return mean
