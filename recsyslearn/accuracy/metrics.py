import numpy as np
import pandas as pd
from abc import ABC
from recsyslearn.errors.errors import RecListTooShortException
from recsyslearn.utils import check_length, check_columns_exist
from recsyslearn.accuracy.utils import ndcg


class AccuracyMetric(ABC):

    """
    Abstract Class for Metrics.
    """

    def __init__(self) -> None:
        return


class NDCG(AccuracyMetric):
    
    """
    NDCG evaluator for recommender systems.
    """

    def evaluate(self, top_n: pd.DataFrame, pos_items: pd.DataFrame, ats: tuple = (5, 10)) -> pd.Series:

        r"""Compute the NDCG@k of a model by using its recommendation list.
        Returns the NDCG averaged over users.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top N recommendations' lists for every user. Columns: ['user', 'item', 'rank'].

        target_df : pd.DataFrame
            Target Interaction dataframe of, i.e., items to be recommended. Columns: ['user', 'item'].

        pos_items : pd.DataFrame
            Relevant items per user. Columns: ['user', 'pos_items'].

        ats: tuple, default (10, )
            The tuple of values at which to evaluate NDCG@k.


        Raises
        ------
        ColumnsNotExistException
            If top_n does not contain columns ('user', 'item', 'rank').

        ColumnsNotExistException
            If target_df does not contain columns ('user', 'item').

        ListTooShortException
            If the top_n list does not contain enough items.


        Return
        ------
        The pd.Series with the NDCG@n averaged over users, in the form ('NDCG@k_0', ..., 'NDCG@k_n')
        """

        check_columns_exist(top_n, ['user', 'item', 'rank'])
        check_columns_exist(pos_items, ['user', 'pos_items'])

        calculable_ats = []

        for k in ats:
            try:
                check_length(top_n, k)
                calculable_ats.append(k)
            except RecListTooShortException as e:
                print(e)
                continue

        top_n = top_n[['user', 'item', 'rank']]
        top_n = top_n[['user', 'item']].groupby('user')['item'].apply(np.asarray).reset_index()
        full_df = top_n.merge(pos_items, on='user')

        columns_to_return = []
        for k in calculable_ats:
            try:
                full_df.loc[:, f'NDCG@{k}'] = full_df.apply(lambda x: ndcg(x['item'][:k], x['pos_items'], at=k), axis=1)
                columns_to_return.append(f'NDCG@{k}')
            except RecListTooShortException as e:
                print(e)
                continue

        return full_df[columns_to_return].mean()
