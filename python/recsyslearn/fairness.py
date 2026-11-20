"""Group fairness metrics.

The metrics follow the probabilistic framework of Amigó, Deldjoo, Mizzaro & Bellogín
(2023). Every recommended ``(user, item)`` pair carries a *benefit*:

* the **exposure** of the item in the list of the user, which decays with the position
  as :math:`1 / \\log_2(1 + position)` (``exposure="logarithmic"``) or is the same at
  every position within the cutoff (``exposure="flat"``);
* the **effectiveness**, i.e. the exposure multiplied by the relevance of the pair,
  when a ``ground_truth`` is given.

The benefit is normalised over all the pairs into a probability distribution, and the
probability of a group of users (or of items) is the benefit its members receive. The
metrics describe that distribution, in bits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from recsyslearn import _core
from recsyslearn._frames import (
    GROUP,
    ITEM,
    USER,
    Recommendations,
    check_choice,
    check_frame,
    check_k,
)
from recsyslearn.errors import InvalidInputError

__all__ = ["entropy", "group_distribution", "kl_divergence", "mutual_information"]

Exposure = Literal["logarithmic", "flat"]
_PROBABILITY = "probability"


def _side(groups: Any, name: str) -> str:
    """Whether ``groups`` assigns users or items to groups."""
    check_frame(groups, name, [GROUP])
    sides = [side for side in (USER, ITEM) if side in groups.columns]
    if len(sides) != 1:
        raise InvalidInputError(
            f"{name} must have either a {USER!r} or an {ITEM!r} column next to {GROUP!r}, "
            f"got the columns {list(groups.columns)}"
        )
    return sides[0]


def _group_codes(
    groups: pd.DataFrame, side: str, name: str, entities: pd.Index, recommended: np.ndarray
) -> tuple[np.ndarray, pd.Index]:
    """Group code of each of the ``entities``, by code, and the group labels.

    Only the ``recommended`` entities need a group. The others (users and items that
    only the ground truth knows) receive no benefit, so their code is a placeholder.
    """
    check_frame(groups, name, [side, GROUP])
    other = ITEM if side == USER else USER
    if other in groups.columns:
        raise InvalidInputError(
            f"{name} must have either a {USER!r} or an {ITEM!r} column next to {GROUP!r}, "
            f"got the columns {list(groups.columns)}"
        )
    if groups[GROUP].isna().any():
        raise InvalidInputError(f"{name} contain missing groups")
    if groups[side].isna().any():
        raise InvalidInputError(f"{name}: {side} identifiers must not be missing")
    members = pd.Index(groups[side], tupleize_cols=False)
    if members.has_duplicates:
        raise InvalidInputError(f"{name} list the same {side} more than once")
    try:
        try:
            codes, labels = pd.factorize(groups[GROUP], sort=True)
        except TypeError:  # labels of mixed types cannot be sorted
            codes, labels = pd.factorize(groups[GROUP])
    except TypeError:
        raise InvalidInputError(f"the group labels of {name} must be hashable") from None

    positions = members.get_indexer(entities)
    ungrouped = int(((positions < 0) & recommended).sum())
    if ungrouped:
        raise InvalidInputError(
            f"{ungrouped} recommended {side}(s) have no group in {name} "
            f"({side} dtypes: {entities.dtype} in recommendations, {members.dtype} in {name})"
        )
    # An entity without a group is not recommended here: any code will do.
    placed = np.zeros(len(positions), dtype=np.uint32)
    grouped = positions >= 0
    placed[grouped] = codes[positions[grouped]]
    return placed, pd.Index(labels, name=GROUP)


def _n_groups(labels: pd.Index | None) -> int:
    """Number of groups to declare to the core for a side.

    At least one when the side is grouped, so that the placeholder code of
    :func:`_group_codes` is valid even for an empty table of groups; nothing is
    recommended in that case, and the core reports that there is no benefit.
    """
    return 0 if labels is None else max(len(labels), 1)


def _distribution(
    recommendations: pd.DataFrame,
    k: int | None,
    exposure: str,
    ground_truth: pd.DataFrame | None,
    user_groups: pd.DataFrame | None = None,
    item_groups: pd.DataFrame | None = None,
    singletons: str | None = None,
) -> tuple[pd.Series | None, pd.Series | None, float]:
    """Marginal distributions over the given groups and their mutual information."""
    k = check_k(k)
    check_choice("exposure", exposure, ("logarithmic", "flat"))
    data = Recommendations(recommendations, ground_truth)

    user_codes = item_codes = user_labels = item_labels = None
    if user_groups is not None:
        user_codes, user_labels = _group_codes(
            user_groups, USER, "user groups", data.users, data.recommended_users
        )
    if item_groups is not None:
        item_codes, item_labels = _group_codes(
            item_groups, ITEM, "item groups", data.items, data.recommended_items
        )

    over_users, over_items, information = _core.benefit_distribution(
        data.lists,
        k,
        exposure,
        data.truth,
        user_codes,
        _n_groups(user_labels),
        item_codes,
        _n_groups(item_labels),
        singletons,
    )
    users = (
        None if user_labels is None else pd.Series(over_users, index=user_labels, name=_PROBABILITY)
    )
    items = (
        None if item_labels is None else pd.Series(over_items, index=item_labels, name=_PROBABILITY)
    )
    return users, items, float(information)


def group_distribution(
    recommendations: pd.DataFrame,
    groups: pd.DataFrame,
    k: int | None = None,
    *,
    exposure: Exposure = "logarithmic",
    ground_truth: pd.DataFrame | None = None,
) -> pd.Series:
    r"""Share of the benefit received by each group of users or items.

    .. math::

        P(g) = \frac{\sum_{(u, i) \in g} \theta(u, i)}{\sum_{(u, i)} \theta(u, i)}

    where the sums run over the recommended pairs (within the top ``k``) and the
    benefit :math:`\theta` is the exposure :math:`\phi(u, i)`, or the effectiveness
    :math:`\phi(u, i) \cdot rel_u(i)` when ``ground_truth`` is given.

    Parameters
    ----------
    recommendations
        Columns ``["user", "item", "rank"]`` (``rank`` optional: without it the order
        of the rows is the ranking).
    groups
        Columns ``["item", "group"]`` to study groups of items (e.g. popular vs niche
        items, providers) or ``["user", "group"]`` for groups of users (e.g. by
        activity or gender). Every recommended item (user) must have exactly one group.
        The functions of :mod:`recsyslearn.segmentation` produce such tables.
    k
        Cutoff; ``None`` uses the whole lists.
    exposure
        ``"logarithmic"``: :math:`\phi = 1 / \log_2(1 + position)`;
        ``"flat"``: :math:`\phi = 1`, i.e. recommendations are simply counted.
    ground_truth
        Held-out interactions, as in :func:`recsyslearn.accuracy.ndcg`. When given,
        only relevant recommendations carry benefit (weighted by their relevance if a
        ``rank`` column is present).

    Returns
    -------
    pandas.Series
        Probability of every group in ``groups``, indexed by group; it sums to 1.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        if ``exposure`` is not one of the two options; if ``groups`` has both a ``user``
        and an ``item`` column or neither, a missing or unhashable group, or the same
        item (user) twice, recommended or not; if a recommended item (user) has no
        group; if a relevance is negative or not finite; or if there is no benefit to
        distribute (no recommendations, or none of them is relevant) or the benefits are
        too large to be added up.

    References
    ----------
    * E. Amigó, Y. Deldjoo, S. Mizzaro and A. Bellogín. *A unifying and general
      account of fairness measurement in recommender systems*. Information Processing
      & Management 60(1), 2023. https://doi.org/10.1016/j.ipm.2022.103115
    * A. Singh and T. Joachims. *Fairness of exposure in rankings*. KDD 2018.
      https://arxiv.org/abs/1802.07281
    """
    side = _side(groups, "groups")
    users, items, _ = _distribution(
        recommendations,
        k,
        exposure,
        ground_truth,
        user_groups=groups if side == USER else None,
        item_groups=groups if side == ITEM else None,
    )
    return users if side == USER else items


def entropy(
    recommendations: pd.DataFrame,
    groups: pd.DataFrame,
    k: int | None = None,
    *,
    exposure: Exposure = "logarithmic",
    ground_truth: pd.DataFrame | None = None,
    normalize: bool = False,
) -> float:
    r"""Shannon entropy of the distribution of the benefit over groups.

    .. math::

        H = -\sum_{g} P(g) \log_2 P(g)

    with :math:`P` as in :func:`group_distribution` and :math:`0 \log 0 = 0`. It is 0
    when a single group receives all the benefit and :math:`\log_2 n` when the
    :math:`n` groups receive the same: higher means more evenly spread.

    Parameters
    ----------
    recommendations, groups, k, exposure, ground_truth
        See :func:`group_distribution`.
    normalize
        Divide by :math:`\log_2 n` so that the result lies in [0, 1], as in Patro et
        al. (2020). It requires at least two groups.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        in the cases listed for :func:`group_distribution`; or if ``normalize`` is set
        and there are fewer than two groups.

    References
    ----------
    * C. E. Shannon. *A mathematical theory of communication*. Bell System Technical
      Journal 27, 1948.
    * G. K. Patro et al. *FairRec: two-sided fairness for personalized recommendations
      in two-sided platforms*. WWW 2020. https://arxiv.org/abs/2002.10764
    * M. Moscati, Y. Deldjoo, G. D. Carparelli and M. Schedl. *Multiobjective
      hyperparameter optimization of recommender systems*. PERSPECTIVES @ RecSys 2023.
      https://ceur-ws.org/Vol-3476/paper3.pdf
    """
    distribution = group_distribution(
        recommendations, groups, k, exposure=exposure, ground_truth=ground_truth
    )
    probabilities = np.ascontiguousarray(distribution.to_numpy(dtype=np.float64))
    return float(_core.entropy(probabilities, bool(normalize)))


def _target(target: Any, labels: pd.Index) -> pd.Series:
    """The target distribution given by the caller, as a Series indexed by group."""
    if isinstance(target, pd.DataFrame):
        check_frame(target, "target", [GROUP, _PROBABILITY])
        target = target.set_index(GROUP)[_PROBABILITY]
    elif isinstance(target, Mapping):
        target = pd.Series(dict(target), dtype=object)
    elif not isinstance(target, pd.Series):
        raise TypeError(
            "target must be a mapping or Series from group to probability, "
            f"or a DataFrame with the columns ['group', 'probability'], got {type(target).__name__}"
        )
    if target.index.has_duplicates:
        raise InvalidInputError("target lists the same group more than once")
    absent = labels.difference(target.index)
    if len(absent):
        raise InvalidInputError(f"target gives no probability to the group(s) {list(absent)}")
    try:
        return target.astype(float)
    except (TypeError, ValueError):
        raise InvalidInputError("the probabilities of target must be numeric") from None


def kl_divergence(
    recommendations: pd.DataFrame,
    groups: pd.DataFrame,
    target: Mapping[Any, float] | pd.Series | pd.DataFrame | None = None,
    k: int | None = None,
    *,
    exposure: Exposure = "logarithmic",
    ground_truth: pd.DataFrame | None = None,
) -> float:
    r"""Inequity: Kullback-Leibler divergence of the benefit from a target distribution.

    .. math::

        D_{KL}(P \,\|\, Q) = \sum_{g} P(g) \log_2 \frac{P(g)}{Q(g)}

    with :math:`P` as in :func:`group_distribution` and :math:`Q` the distribution
    regarded as fair. It is 0 when the benefit is distributed exactly as the target
    and grows as it departs from it; it is infinite when a group that the target
    excludes (:math:`Q(g) = 0`) receives some benefit.

    Parameters
    ----------
    recommendations, groups, k, exposure, ground_truth
        See :func:`group_distribution`.
    target
        The fair distribution :math:`Q`: a mapping or Series from group to
        probability, or a DataFrame with the columns ``["group", "probability"]``. It
        must sum to 1 (within :math:`10^{-6}`, to allow for rounded shares) and cover
        every group in ``groups``; the order of its rows does not matter. ``None`` means parity: the
        same probability for every group in ``groups``. Other common choices are the
        share of the catalog or of the training interactions of each group.

    Raises
    ------
    TypeError
        If a table is not a DataFrame, or ``target`` is not a mapping, a Series or a
        DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        in the cases listed for :func:`group_distribution`; or if ``target`` lists a
        group twice, omits a group of ``groups``, or holds probabilities that are not
        numbers, are negative or not finite, or do not sum to 1 within :math:`10^{-6}`.

    References
    ----------
    * E. Amigó, Y. Deldjoo, S. Mizzaro and A. Bellogín. *A unifying and general
      account of fairness measurement in recommender systems* (Eq. 1). Information
      Processing & Management 60(1), 2023. https://doi.org/10.1016/j.ipm.2022.103115
    * M. Moscati, Y. Deldjoo, G. D. Carparelli and M. Schedl. *Multiobjective
      hyperparameter optimization of recommender systems*. PERSPECTIVES @ RecSys 2023.
      https://ceur-ws.org/Vol-3476/paper3.pdf
    """
    distribution = group_distribution(
        recommendations, groups, k, exposure=exposure, ground_truth=ground_truth
    )
    if target is None:
        probabilities = np.ascontiguousarray(distribution.to_numpy(dtype=np.float64))
        return float(_core.kl_divergence(probabilities, None))
    target = _target(target, distribution.index)
    # The groups of the distribution in their own order, then those that only the
    # target knows (they receive no benefit): the order of the target must not matter.
    groups = distribution.index.append(target.index.difference(distribution.index))
    return float(
        _core.kl_divergence(
            np.ascontiguousarray(distribution.reindex(groups, fill_value=0.0).to_numpy(np.float64)),
            np.ascontiguousarray(target.reindex(groups).to_numpy(np.float64)),
        )
    )


def mutual_information(
    recommendations: pd.DataFrame,
    *,
    user_groups: pd.DataFrame | None = None,
    item_groups: pd.DataFrame | None = None,
    k: int | None = None,
    exposure: Exposure = "logarithmic",
    ground_truth: pd.DataFrame | None = None,
) -> float:
    r"""Dependence: mutual information between who is served and what is recommended.

    .. math::

        I(X; Y) = \sum_{x, y} P(x, y) \log_2 \frac{P(x, y)}{P(x) P(y)}

    where :math:`P(x, y)` is the share of the benefit carried by the recommendations
    of items of group :math:`y` to users of group :math:`x`. It is 0 when the groups of
    items a user is shown do not depend on the group of the user, and it is bounded by
    the entropy of either marginal.

    Parameters
    ----------
    recommendations, k, exposure, ground_truth
        See :func:`group_distribution`.
    user_groups
        Columns ``["user", "group"]``. If omitted, every user is a group of their own:
        the result is the dependence between individual users and item groups.
    item_groups
        Columns ``["item", "group"]``. If omitted, every item is a group of its own.
        At least one of ``user_groups`` and ``item_groups`` is required.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        in the cases listed for :func:`group_distribution`, for either table of groups;
        or if neither ``user_groups`` nor ``item_groups`` is given.

    References
    ----------
    * E. Amigó, Y. Deldjoo, S. Mizzaro and A. Bellogín. *A unifying and general
      account of fairness measurement in recommender systems* (Eq. 2). Information
      Processing & Management 60(1), 2023. https://doi.org/10.1016/j.ipm.2022.103115
    """
    if user_groups is None and item_groups is None:
        raise InvalidInputError("at least one of user_groups and item_groups is required")
    singletons = USER if user_groups is None else ITEM if item_groups is None else None
    _, _, information = _distribution(
        recommendations, k, exposure, ground_truth, user_groups, item_groups, singletons
    )
    return information
