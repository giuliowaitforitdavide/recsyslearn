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
