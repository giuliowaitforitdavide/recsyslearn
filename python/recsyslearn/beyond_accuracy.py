"""Beyond-accuracy metrics: coverage of the catalog and novelty of the items."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from recsyslearn import _core
from recsyslearn._frames import ITEM, RANK, USER, check_frame, check_k, encode, values
from recsyslearn.errors import InvalidInputError

__all__ = ["coverage", "novelty"]


def coverage(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame | Iterable[Any],
    k: int | None = None,
) -> float:
    r"""Catalog coverage: the share of the catalog recommended to at least one user.

    .. math::

        \mathrm{coverage@}k = \frac{\left| \bigcup_{u} L_u \right|}{|I|}

    where :math:`L_u` is the top-``k`` list of user :math:`u` and :math:`I` the
    catalog, i.e. all the items that could be recommended.

    Parameters
    ----------
    recommendations
        Columns ``["user", "item", "rank"]`` (``rank`` optional, see
        :func:`recsyslearn.accuracy.ndcg`).
    catalog
        The items available for recommendation: any collection of item identifiers,
        or a DataFrame with an ``item`` column (e.g. the training interactions).
        Repeated identifiers count once.
    k
        Cutoff; ``None`` uses the whole lists.

    Raises
    ------
    TypeError
        If a table is not a DataFrame, or the catalog is not a collection of items.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        if the catalog is empty, is not a flat collection of hashable identifiers or
        contains a missing identifier; or if a recommended item is not part of it.

    References
    ----------
    * M. Ge, C. Delgado-Battenfeld and D. Jannach. *Beyond accuracy: evaluating
      recommender systems by coverage and serendipity*. RecSys 2010.
      https://doi.org/10.1145/1864708.1864761
    """
    k = check_k(k)
    check_frame(recommendations, "recommendations", [USER, ITEM], [RANK])
    if isinstance(catalog, pd.DataFrame):
        check_frame(catalog, "catalog", [ITEM])
        catalog = catalog[ITEM]
    elif not isinstance(catalog, (pd.Series, pd.Index, np.ndarray)):
        if isinstance(catalog, (str, bytes)) or not isinstance(catalog, Iterable):
            raise TypeError(
                "catalog must be a collection of item identifiers or a DataFrame with an "
                f"'item' column, got {type(catalog).__name__}"
            )
        # One element per item, whatever an item is: a tuple is an identifier, not a row.
        catalog = pd.Series(list(catalog), dtype=object).infer_objects()
    if np.ndim(catalog) != 1:
        raise InvalidInputError("catalog must be a flat (one-dimensional) collection of items")
    try:
        distinct = pd.unique(pd.Series(catalog, dtype=None if len(catalog) else object))
    except TypeError:
        raise InvalidInputError(
            "catalog must be a flat collection of hashable item identifiers"
        ) from None
    items = pd.Index(distinct, tupleize_cols=False)
    if items.empty:
        raise InvalidInputError("the catalog is empty")
    if items.isna().any():
        raise InvalidInputError("the catalog contains missing item identifiers")

    positions = items.get_indexer(pd.Index(recommendations[ITEM], tupleize_cols=False))
    outside = int((positions < 0).sum())
    if outside:
        raise InvalidInputError(
            f"{outside} recommendation(s) refer to items that are not in the catalog "
            f"(item dtypes: {recommendations[ITEM].dtype} in recommendations, "
            f"{items.dtype} in catalog)"
        )
    (user_codes,), users = encode(recommendations[USER])
    item_codes = np.ascontiguousarray(positions, dtype=np.uint32)
    ranks = values(recommendations, RANK, "recommendations")
    lists = _core.RankedLists(user_codes, item_codes, ranks, len(users))
    return float(_core.catalog_coverage(lists, k, len(items)))


def novelty(
    recommendations: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int | None = None,
    *,
    per_user: bool = False,
) -> float | pd.Series:
    r"""Novelty: the mean self-information of the recommended items.

    The less popular an item, the more surprising it is to see it recommended. With
    :math:`pop_i = |U_i| / |U|` the fraction of users who interacted with item
    :math:`i`,

    .. math::

        \mathrm{novelty@}k = \frac{1}{|U_L|} \sum_{u \in U_L} \frac{1}{|L_u|}
            \sum_{i \in L_u} -\log_2 pop_i

    where :math:`U_L` are the users who received recommendations and :math:`L_u`
    their top-``k`` lists. The result is in bits: 0 when every recommended item is
    known by all users, and at most :math:`\log_2 |U|`.

    Parameters
    ----------
    recommendations
        Columns ``["user", "item", "rank"]`` (``rank`` optional, see
        :func:`recsyslearn.accuracy.ndcg`).
    interactions
        The interactions the popularity is measured on, typically the training set.
        Columns ``["user", "item"]``; a ``rank`` column (explicit feedback) is
        ignored. A user who interacted with an item several times counts once.
    k
        Cutoff; ``None`` uses the whole lists.
    per_user
        Return the novelty of every user's list instead of their mean.

    Notes
    -----
    The formula diverges for a recommended item that nobody interacted with, and the
    literature does not say how to treat it. Such an item is assigned the
    self-information of an item with a single user, :math:`\log_2 |U|`.

    Raises
    ------
    TypeError
        If a table is not a DataFrame.
    MissingColumnsError
        If a table lacks a column it needs.
    InvalidInputError
        If a table or ``k`` is invalid in one of the ways listed in :ref:`input-errors`;
        or if there are no recommendations or no interactions.

    References
    ----------
    * T. Zhou et al. *Solving the apparent diversity-accuracy dilemma of recommender
      systems*. PNAS 107(10), 2010. https://doi.org/10.1073/pnas.1000488107
    * M. Schedl et al. *Current challenges and visions in music recommender systems
      research* (Eq. 12). IJMIR 7, 2018. https://doi.org/10.1007/s13735-018-0154-2
    * S. Vargas and P. Castells. *Rank and relevance in novelty and diversity metrics
      for recommender systems*. RecSys 2011. https://doi.org/10.1145/2043932.2043955
    """
    k = check_k(k)
    check_frame(recommendations, "recommendations", [USER, ITEM], [RANK])
    check_frame(interactions, "interactions", [USER, ITEM])
    if recommendations.empty:
        raise InvalidInputError("there are no recommendations to evaluate")
    if interactions.empty:
        raise InvalidInputError("interactions contain no users")

    (known_items, recommended_items), items = encode(interactions[ITEM], recommendations[ITEM])
    (known_users,), population = encode(interactions[USER])
    # Recommended items without interactions have degree 0: the core treats them as unseen.
    degrees = _core.distinct_counts(known_items, known_users, len(items))

    (user_codes,), users = encode(recommendations[USER])
    ranks = values(recommendations, RANK, "recommendations")
    lists = _core.RankedLists(user_codes, recommended_items, ranks, len(users))

    scores = _core.novelty(lists, k, degrees, len(population))
    if per_user:
        return pd.Series(scores, index=users, name="novelty")
    return float(_core.mean_over_users(scores))
