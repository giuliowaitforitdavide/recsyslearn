import numpy as np
import pandas as pd

from recsyslearn.utils import check_columns_exist


def exp_matrix(top_n: pd.DataFrame) -> pd.DataFrame:
    """
    Compute exposure matrix for given recommendation lists.

    :param top_n: Recommendation lists per user in the form (user, item, rank, group).
    :type top_n: pd.DataFrame
    :raises ColumnsNotExistException: If top_n is not in the form (user, item, rank, group).
    :return: A new DataFrame with computed exposure in the rank column; top_n is not modified.
    :rtype: pd.DataFrame
    """

    check_columns_exist(top_n, ["user", "item", "rank", "group"])

    return _with_rank(top_n, 1 / np.log2(1 + top_n["rank"]))


def prob_matrix(top_n: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probability distribution matrix for given recommendation lists.

    :param top_n: Recommendation lists per user in the form (user, item, rank, group).
    :type top_n: pd.DataFrame
    :raises ColumnsNotExistException: If top_n is not in the form (user, item, rank, group).
    :return: A new DataFrame with the probability distribution in the rank column; top_n is not modified.
    :rtype: pd.DataFrame
    """

    check_columns_exist(top_n, ["user", "item", "rank", "group"])

    return _with_rank(top_n, top_n["rank"] / top_n["rank"].sum())


def _with_rank(top_n: pd.DataFrame, rank: pd.Series) -> pd.DataFrame:
    """
    Return a new DataFrame equal to top_n with the rank column replaced, leaving top_n unchanged.

    Only the new rank column is allocated: the other columns are shared with top_n through a
    shallow copy, so that large DataFrames are not duplicated in memory.

    :param top_n: Recommendation lists per user with a rank column.
    :type top_n: pd.DataFrame
    :param rank: The new values of the rank column, aligned with top_n.
    :type rank: pd.Series
    :return: A new DataFrame with the given rank column.
    :rtype: pd.DataFrame
    """

    result = top_n.copy(deep=False)
    result["rank"] = rank
    return result


def eff_matrix(top_n: pd.DataFrame, rel_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute effectiveness matrix for given recommendation lists.

    :param top_n: Recommendation lists per user in the form (user, item, rank, group).
    :type top_n: pd.DataFrame
    :param rel_matrix: Dataframe containing relevant items for every user.
    :type rel_matrix: pd.DataFrame
    :raises ColumnsNotExistException: If top_n header is not in the form (user, item, rank, group)
        of if rel_matrix header is not in the form (user, item, rank).
    :return: The DataFrame with computed effectiveness.
    :rtype: pd.DataFrame
    """

    check_columns_exist(top_n, ["user", "item", "rank", "group"])
    check_columns_exist(rel_matrix, ["user", "item", "rank", "group"])

    top_n = exp_matrix(top_n)
    top_n = top_n.merge(rel_matrix, on=["user", "item", "group"], how="outer")
    top_n.loc[:, ["rank_x", "rank_y"]] = top_n.loc[:, ["rank_x", "rank_y"]].fillna(0)
    top_n["rank"] = top_n["rank_x"] * top_n["rank_y"]
    return top_n[["user", "item", "rank", "group"]]
