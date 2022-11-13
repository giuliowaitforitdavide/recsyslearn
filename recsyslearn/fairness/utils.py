import pandas as pd
import numpy as np
from recsyslearn.utils import check_columns_exist


def exp_matrix(top_n: pd.DataFrame) -> pd.DataFrame:
    """
    Compute exposure matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
        Recommendation lists per user in the form (user, item, rank, group).


    Raises
    ------
    ColumnsNotExistException
        If top_n is not in the form (user, item, rank, group).


    Return
    ------
    top_n : pd.DataFrame
        The DataFrame with computed exposure.
    """

    check_columns_exist(top_n, ["user", "item", "rank", "group"])

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
    ColumnsNotExistException
        If top_n is not in the form (user, item, rank, group).
    """

    check_columns_exist(top_n, ["user", "item", "rank", "group"])

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


    Raises
    ------
    ColumnsNotExistException
        If top_n header is not in the form (user, item, rank, group).
        If rel_matrix header is not in the form (user, item, rank).
    """

    check_columns_exist(top_n, ["user", "item", "rank", "group"])
    check_columns_exist(rel_matrix, ["user", "item", "rank", "group"])

    top_n = exp_matrix(top_n)
    top_n = top_n.merge(rel_matrix, on=["user", "item", "group"], how="outer")
    top_n.loc[:, ["rank_x", "rank_y"]] = top_n.loc[:,
                                                   ["rank_x", "rank_y"]].fillna(0)
    top_n["rank"] = top_n["rank_x"] * top_n["rank_y"]
    return top_n[["user", "item", "rank", "group"]]
