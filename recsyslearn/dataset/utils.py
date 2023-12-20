import numpy as np
import pandas as pd

from recsyslearn.utils import check_columns_exist


def find_relevant_items(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find relevant items for every user in the dataset.

    :param target_df: Target Interaction dataframe of, i.e., items to be recommended. Columns: ['user', 'item'].
    :type target_df: pd.DataFrame
    :raises ColumnsNotExistException: If target_df does not contain columns ('user', 'item').
    :return: The DataFrame containing all the relevant items per user in the form ('user', 'pos_items').
    :rtype: pd.DataFrame
    """

    check_columns_exist(target_df, ["user", "item"])
    target_df = target_df[["user", "item"]]
    pos_items = target_df.groupby("user")["item"].apply(np.asarray).reset_index()
    pos_items.columns = ["user", "pos_items"]
    return pos_items
