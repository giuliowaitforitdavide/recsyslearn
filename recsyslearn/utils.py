import pandas as pd

from recsyslearn.errors.errors import ColumnsNotExistException


def check_columns_exist(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Raise ColumnsNotExistException if pd.Dataframe does not contain the expected columns.

    :param df: Input that should be tested.
    :type df: pd.DataFrame
    :param columns: pd.DataFrame columns that should be contained.
    :type columns: list
    :raises ColumnsNotExistException: If input does not contained expected columns.
    :return: The DataFrame with the expected columns and types.
    :rtype: pd.DataFrame
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

    return df.astype({col: dtypes[col] for col in df.columns if col in dtypes.keys()})
