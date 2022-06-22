import pandas as pd
import numpy as np

from recsyslearn.errors import ColumnsNotMatchException


def test_pattern(input: pd.DataFrame, pattern: list) -> None:

    '''
    Raise ColumnsNotMatchException if pd.Dataframe header is not as expected.

    Parameters
    ----------
    input : pd.DataFrame
      Input that should be tested.

    pattern : list
      pd.DataFrame column pattern that should be matched. (a header).


    Raise
    ------
    ColumnsNotMatchException
      If input as not expected.

    
    Return
    ------
    Converted columns within the same DataFrame.
    '''

    if list(input.columns) != pattern:
        raise ColumnsNotMatchException(pattern)

    dtypes = {
      'user': str,
      'item': str,
      'rank': float,
      'group': str,
      'target_representation': float
      }

    return input.astype({col: dtypes[col] for col in input.columns if col in dtypes.keys()})
    


def exp_matrix(top_n: pd.DataFrame) -> pd.DataFrame:

    '''
    Compute exposure matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
      	Recommendation lists per user in the form (user, item, rank, group).


    Raise
    -----
    ColumnsNotMatchException
      	If top_n is not in the from (user, item, rank, group).


    Return
    ------
    top_n : pd.DataFrame
      	The DataFrame with computed exposure.
    '''

    test_pattern(top_n, ['user', 'item', 'rank', 'group'])

    top_n['rank'] = 1 / np.log2(1 + top_n['rank'])
    return top_n


def prob_matrix(top_n: pd.DataFrame) -> pd.DataFrame:

    '''
    Compute probability distribution matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
      	Recommendation lists per user in the form (user, item, rank, group).


    Return
    ------
    top_n : pd.DataFrame
      	The DataFrame with computed probability distribution.


    Raise
    -----
    ColumnsNotMatchException
      	If top_n is not in the from (user, item, rank, group).
    '''

    test_pattern(top_n, ['user', 'item', 'rank', 'group'])
    
    top_n['rank'] = top_n['rank'] / top_n['rank'].sum()
    return top_n


def eff_matrix(top_n: pd.DataFrame, rel_matrix: pd.DataFrame) -> pd.DataFrame:

    '''
    Compute effectiveness matrix for given recommendation lists.

    Parameters
    ----------
    top_n : pd.DataFrame
      	Recommendation lists per user in the form (user, item, rank, group).


    Return
    ------
    top_n : pd.DataFrame
      	The DataFrame with computed effectiveness.


    Raise
    -----
    ColumnsNotMatchException
      	If top_n header is not in the from (user, item, rank, group).
      	If rel_matrix header is not in the form (user, item, rank).
    '''

    test_pattern(top_n, ['user', 'item', 'rank', 'group'])
    test_pattern(rel_matrix, ['user', 'item', 'rank', 'group'])

    top_n = exp_matrix(top_n)
    top_n = top_n.merge(rel_matrix, on=['user', 'item', 'group'], how='outer')
    top_n.loc[:, ['rank_x', 'rank_y']] = top_n.loc[:, ['rank_x', 'rank_y']].fillna(0)
    top_n['rank'] = top_n['rank_x'] * top_n['rank_y']
    return top_n[['user', 'item', 'rank', 'group']]
