import pandas as pd
from recsyslearn.errors.errors import ColumnsNotExistException, RecListTooShortException


def check_columns_exist(df: pd.DataFrame, columns: list) -> pd.DataFrame:
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


def check_length(df: pd.DataFrame, k: int) -> None:

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

    check_columns_exist(df, ['user', 'item', 'rank'])
    rec_list_length = df['rank'].max()

    if rec_list_length < k:
        raise RecListTooShortException(k)
