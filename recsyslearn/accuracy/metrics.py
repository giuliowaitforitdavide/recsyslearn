import numpy as np
import pandas as pd
from abc import ABC
from recsyslearn.errors.errors import RecListTooShortException
from recsyslearn.utils import check_columns_exist
import warnings


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

    @classmethod
    def __dcg(cls, scores: pd.DataFrame) -> float:
        r"""Compute DCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation.

        Parameters
        ----------
        scores : pd.DataFrame
            Scores of the recommended items.


        Return
        ------
        The DCG of the single user.
        """

        return np.sum(
            np.divide(
                np.power(2, scores) - 1,
                np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)
            ),
            dtype=np.float32)

    @classmethod
    def __ndcg(cls, ranked_list: pd.DataFrame, pos_items: pd.DataFrame, relevance: pd.DataFrame = None, at: int = None) -> float:
        r"""Compute NDCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation.


        Parameters
        ----------
        ranked_list : pd.DataFrame
            Recommendation list of length \"at\".

        pos_items : pd.DataFrame
            Relevant items per user. Columns: ['user', 'pos_items'].

        relevance: pd.DataFrame
            Relevance score associated with pos_items, default None.

        at: int
            Length of reclist to be evaluated, default None.


        Return
        ------
        The NDCG@at of the single user.
        """

        if relevance is None:
            relevance = np.ones_like(pos_items, dtype=np.int32)
        assert len(relevance) == pos_items.shape[0]

        it2rel = {it: r for it, r in zip(pos_items, relevance)}

        rank_scores = np.asarray([it2rel.get(it, 0.0)
                                 for it in ranked_list[:at]], dtype=np.float32)
        ideal_dcg = cls.__dcg(np.sort(relevance)[::-1][:at])
        rank_dcg = cls.__dcg(rank_scores)

        if rank_dcg == 0:
            return 0

        ndcg_ = rank_dcg / ideal_dcg

        return ndcg_

    @classmethod
    def evaluate(cls, top_n: pd.DataFrame, pos_items: pd.DataFrame, ats: tuple = (5, 10)) -> pd.Series:
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

        ats: tuple, default (5, 10)
            The tuple of values at which to evaluate NDCG@k.


        Raises
        ------
        ColumnsNotExistException
            If top_n does not contain columns ('user', 'item', 'rank').

        ColumnsNotExistException
            If target_df does not contain columns ('user', 'item').

        RecListTooShortException
            If the top_n list does not contain enough items.


        Return
        ------
        The pd.Series with the NDCG@n averaged over users, in the form ('NDCG@k_0', ..., 'NDCG@k_n')
        """

        check_columns_exist(top_n, ['user', 'item', 'rank'])
        check_columns_exist(pos_items, ['user', 'pos_items'])

        min_calculable_at = top_n.groupby('user').size().min()
        calculable_ats = [k for k in ats if k <= min_calculable_at]

        if len(calculable_ats) == 0:
            raise RecListTooShortException(ats)

        if len(calculable_ats) != len(ats):
            non_calulable_ats = list(set(calculable_ats) - set(ats))
            warnings.warn(non_calulable_ats.join(
                ' ') + ' ats won\'t be calculated')

        top_n = top_n[['user', 'item', 'rank']]
        top_n = top_n[['user', 'item']].groupby(
            'user')['item'].apply(np.asarray).reset_index()
        full_df = top_n.merge(pos_items, on='user')

        for k in calculable_ats:
            full_df.loc[:, f'NDCG@{k}'] = full_df.apply(
                lambda x: cls.__ndcg(x['item'][:k], x['pos_items'], at=k), axis=1)

        cols_to_be_returned = [
            col for col in full_df.columns if col not in ['item', 'pos_items']]

        return full_df.loc[:, cols_to_be_returned]
