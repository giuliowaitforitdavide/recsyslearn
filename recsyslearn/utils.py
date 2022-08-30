import pandas as pd
import numpy as np
from pandas import DataFrame

from recsyslearn.errors import ColumnsNotMatchException, ColumnsNotExistException, ListTooShortException


def test_pattern(df: pd.DataFrame, pattern: list) -> DataFrame:
    """
    Raise ColumnsNotMatchException if pd.Dataframe header is not as expected.

    Parameters
    ----------
    df : pd.DataFrame
      Input that should be tested.

    pattern : list
      pd.DataFrame column pattern that should be matched. (a header).


    Raises
    ------
    ColumnsNotMatchException
      If input as not expected.


    Return
    ------
    Converted columns within the same DataFrame.
    """

    if list(df.columns) != pattern:
        raise ColumnsNotMatchException(pattern)

    dtypes = {
        "user": str,
        "item": str,
        "rank": float,
        "group": str,
        "target_representation": float,
    }

    return df.astype(
        {col: dtypes[col] for col in df.columns if col in dtypes.keys()}
    )


def test_columns_exist(df: pd.DataFrame, columns: list) -> None:
    """
    Raise ColumnsNotExistException if pd.Dataframe does not contain the expected columns.

    Parameters
    ----------
    df : pd.DataFrame
      Input that should be tested.

    columns : list
      pd.DataFrame columns that should be contained.


    Raises
    ------
    ColumnsNotExistException
      If input does not contained expected columns.
    """
    if not set(columns).issubset(set(df.columns)):
        raise ColumnsNotExistException(columns)


def test_length(df: pd.DataFrame, k: int) -> None:
    """
    Raise ListTooShortException if pd.Dataframe does not contain enough recommendation
    per user to compute NDCG@k, with k in ats.

    Parameters
    ----------
    df : pd.DataFrame
      The recommendation dataframe containing the ranking lists per user.
      Should contain ['user', 'item', 'rank'] columns.

    k : int
        The value at which to compute the metric. The recommendation list
        per user should be longer than k.


    Raises
    ------
    ColumnsNotExistException
      If input does not contained expected columns.
    ListTooShortException
      If the recommendation list is too short to compute the metric at k.
    """
    test_columns_exist(df, ['user', 'item', 'rank'])
    rec_list_length = df['rank'].max()

    if rec_list_length < k:
        raise ListTooShortException(k)


def exp_matrix(top_n: pd.DataFrame) -> pd.DataFrame:
    """
    Compute exposure matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
        Recommendation lists per user in the form (user, item, rank, group).


    Raises
    -----
    ColumnsNotMatchException
        If top_n is not in the form (user, item, rank, group).


    Return
    ------
    top_n : pd.DataFrame
        The DataFrame with computed exposure.
    """

    test_pattern(top_n, ["user", "item", "rank", "group"])

    top_n["rank"] = 1 / np.log2(1 + top_n["rank"])
    return top_n


def prob_matrix(top_n: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probability distribution matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
        Recommendation lists per user in the form (user, item, rank, group).


    Return
    ------
    top_n : pd.DataFrame
        The DataFrame with computed probability distribution.


    Raises
    ------
    ColumnsNotMatchException
        If top_n is not in the form (user, item, rank, group).
    """

    test_pattern(top_n, ["user", "item", "rank", "group"])

    top_n["rank"] = top_n["rank"] / top_n["rank"].sum()
    return top_n


def eff_matrix(top_n: pd.DataFrame, rel_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute effectiveness matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
        Recommendation lists per user in the form (user, item, rank, group).

    rel_matrix : pd.DataFrame
        Dataframe containing relevant items for every user.


    Return
    ------
    top_n : pd.DataFrame
        The DataFrame with computed effectiveness.


    Raise
    -----
    ColumnsNotMatchException
        If top_n header is not in the from (user, item, rank, group).
        If rel_matrix header is not in the form (user, item, rank).
    """

    test_pattern(top_n, ["user", "item", "rank", "group"])
    test_pattern(rel_matrix, ["user", "item", "rank", "group"])

    top_n = exp_matrix(top_n)
    top_n = top_n.merge(rel_matrix, on=["user", "item", "group"], how="outer")
    top_n.loc[:, ["rank_x", "rank_y"]] = top_n.loc[:, ["rank_x", "rank_y"]].fillna(0)
    top_n["rank"] = top_n["rank_x"] * top_n["rank_y"]
    return top_n[["user", "item", "rank", "group"]]


def ndcg(ranked_list, pos_items, relevance=None, at=None):
    """ Compute NDCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. """

    if relevance is None:
        relevance = np.ones_like(pos_items, dtype=np.int32)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)
    # IDCG has all relevances to 1, up to the number of items in the test set
    # Fixed bug in PoliMi code.
    ideal_dcg = dcg(np.sort(relevance)[::-1][:at])
    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)
    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


def dcg(scores):
    """ Compute DCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. """

    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)
