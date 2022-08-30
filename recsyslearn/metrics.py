import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from recsyslearn.errors import FlagNotValidException, ListTooShortException
from recsyslearn.utils import eff_matrix, exp_matrix, test_pattern, prob_matrix, ndcg, test_columns_exist, test_length


class Metric(ABC):
    """
    Abstract Class for Metrics.
    """

    def __init__(self) -> None:
        return


class Coverage(Metric):
    """
    Coverage evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    """

    def evaluate(self, top_n: pd.DataFrame, items: list) -> float:
        """
        Compute the coverage of a model by using its recommendation list.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top-N recommendations' lists for every user with items or users already segmented.

        items : list or array-like
            List of items in the dataset.


        Raises
        ------
        ColumnsNotMatchException
            If top_n not in the form ('user', 'item', 'rank', 'group').


        Return
        ------
        The computed coverage.
        """

        test_pattern(top_n, ['user', 'item', 'rank'])
        return len(top_n.item.unique().tolist()) / len(items)


class Novelty(Metric):
    """
    Novelty evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    where popularity is defined in terms of the segmentation of the item groups
    (e.g.  short head -> 3
           mid tail   -> 2
           long tail  -> 1)
    or in terms of percentage of user-item interactions.
    """

    def evaluate(self, top_n: pd.DataFrame, popularity_definition='group') -> float:
        """
        Compute the novelty of a model by using its recommendation list and the segmented item groups.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top-N recommendations' lists for every user with items or users already segmented.
        popularity_definition: str
            Either 'group' or 'percentage', to choose whether popularity is computed in terms of
            segmenting items/users according to the distribution of user-item interactions
            or if it is defined as the percentage of user-item interactions.

        Raises
        ------
        ColumnsNotMatchException
            If top_n not in the form ('user', 'item', 'rank', popularity_definition).


        Return
        ------
        The computed novelty.
        """

        test_pattern(top_n, ['user', 'item', 'rank', popularity_definition])

        top_n.loc[:, popularity_definition] = pd.to_numeric(top_n.loc[:, popularity_definition])
        top_n = top_n.groupby('user')[popularity_definition].apply(lambda x: np.mean(- np.log2(x)))
        return top_n.mean()


class Entropy(Metric):
    """
    Entropy evaluator for recommender systems.
    """

    def evaluate(self, top_n: pd.DataFrame, rel_matrix: pd.DataFrame = None) -> float:
        """
        Compute the entropy of a model by using its recommendation list.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top N recommendations' lists for every user with items or users already segmented.

        rel_matrix : pd.DataFrame, default None
            Relevant items for users. It could be, for example, the items with a rating >= threshold.


        Raises
        ------
        ColumnsNotMatchException
            If top_n not in the form ('user', 'item', 'rank', 'group').


        Return
        ------
        The computed entropy.
        """

        test_pattern(top_n, ['user', 'item', 'rank', 'group'])

        if rel_matrix is not None:
            top_n = eff_matrix(top_n, rel_matrix)

        top_n = prob_matrix(top_n)

        top_n = top_n[['group', 'rank']].groupby('group', as_index=False).sum()
        top_n['rank'] = top_n['rank'] * np.log2(top_n['rank'])
        return - top_n['rank'].sum()


class KullbackLeibler(Metric):
    """
    Kullback-Leibler divergence evaluator for recommender systems.
    """

    def evaluate(self, top_n: pd.DataFrame, target_representation: pd.DataFrame,
                 rel_matrix: pd.DataFrame = None) -> float:

        """
        Compute the Kullback-Leibler divergence of a model, for a given target representation, by using its recommendation list.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top N recommendations' lists for every user with items or users already segmented.

        target_representation : pd.DataFrame
            The target representation desired for each group.

        rel_matrix : pd.DataFrame, default None
            Relevant items for users. It could be, for example, the items with a rating >= threshold.


        Raises
        ------
        ColumnsNotMatchException
            If top_n not in the form ('user', 'item', 'rank', 'group') or if target_representation not in the form ('group', 'target_representation').


        Return
        ------
        The computed KL Divergence for the given target representation.
        """

        test_pattern(top_n, ['user', 'item', 'rank', 'group'])
        test_pattern(target_representation, ['group', 'target_representation'])

        if rel_matrix is not None:
            top_n = eff_matrix(top_n, rel_matrix)
        else:
            top_n = exp_matrix(top_n)

        top_n = prob_matrix(top_n)
        top_n = top_n[['group', 'rank']].groupby('group', as_index=False).sum()
        top_n = top_n.merge(target_representation, on='group')
        top_n['rank'] = top_n['rank'] * np.log2(top_n['rank'] / top_n['target_representation'])
        return top_n['rank'].sum()


class MutualInformation(Metric):
    """
    Mutual Information evaluator for recommender systems.
    """

    def evaluate(self, top_n: pd.DataFrame, flag: str, rel_matrix: pd.DataFrame = None) -> float:

        """
        Compute the Mutual Information of a model by using its recommendation list.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top N recommendations' lists for every user with items or users already segmented.

        flag : str
            Which actor of the recommendation scenario has been segmented (i.e. user).

        rel_matrix : pd.DataFrame, default None
            Relevant items for users. It could be, for example, the items with a rating >= threshold.


        Raises
        ------
        ColumnsNotMatchException
            If top_n not in the form ('user', 'item', 'rank', 'group').

        FlagNotValidException
            If flag is not valid.


        Return
        ------
        The computed Mutual Information.
        """

        test_pattern(top_n, ['user', 'item', 'rank', 'group'])

        not_flagged = {'user': 'item', 'item': 'user'}

        if flag not in list(not_flagged.keys()):
            raise FlagNotValidException()

        if rel_matrix is not None:
            top_n = eff_matrix(top_n, rel_matrix)
        else:
            top_n = exp_matrix(top_n)

        top_n = prob_matrix(top_n)

        P_xy = top_n[[not_flagged[flag], 'group', 'rank']].groupby([not_flagged[flag], 'group'], as_index=False).sum()
        P_xP_y = top_n[[not_flagged[flag], 'group', 'rank']].groupby(not_flagged[flag], as_index=False).sum()
        P_xP_y = P_xy[[not_flagged[flag], 'group']].merge(P_xP_y, on=not_flagged[flag])
        tmp = top_n[['group', 'rank']].groupby('group', as_index=False).sum()
        P_xP_y = P_xP_y.merge(tmp, on=['group'])
        P_xP_y['rank'] = P_xP_y['rank_x'] * P_xP_y['rank_y']
        tmp = P_xP_y[[not_flagged[flag], 'group', 'rank']].merge(P_xy, on=[not_flagged[flag], 'group'])
        tmp['rank'] = tmp['rank_y'] * np.log2(tmp['rank_y'] / tmp['rank_x'])
        return tmp['rank'].sum()


class NDCG(Metric):
    """
    NDCG evaluator for recommender systems.
    """

    def evaluate(self, top_n: pd.DataFrame, target_df: pd.DataFrame, ats: tuple = (5, 10)) -> pd.Series:
        """
        Compute the NDCG@k of a model by using its recommendation list.
        Returns the NDCG averaged over users.


        Parameters
        ----------
        top_n : pd.DataFrame
            Top N recommendations' lists for every user. Columns: ['user', 'item', 'rank'].

        target_df : pd.DataFrame
            Target Interaction dataframe of, i.e., items to be recommended. Columns: [user, item].

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

        test_columns_exist(top_n, ['user', 'item', 'rank'])
        test_columns_exist(target_df, ['user', 'item'])

        top_n = top_n[['user', 'item', 'rank']]
        target_df = target_df[['user', 'item']]

        calculable_ats = []

        for k in ats:
            try:
                test_length(top_n, k)
                calculable_ats += [k]
            except ListTooShortException as e:
                print(e)
                continue
        calculable_ats = tuple(calculable_ats)

        # Get for each user the items they interacted with
        pos_items = target_df.groupby('user')['item'].apply(np.asarray).reset_index()
        pos_items.columns = ['user', 'pos_items']

        # Convert the ranked lists to lists instead of entries of the df
        top_n = top_n[['user', 'item']].groupby('user')['item'].apply(np.asarray).reset_index()
        full_df = top_n.merge(pos_items, on='user')

        columns_to_return = []
        for k in calculable_ats:
            try:
                full_df.loc[:, f'NDCG@{k}'] = full_df.apply(lambda x: ndcg(x['item'][:k], x['pos_items'], at=k), axis=1)
                columns_to_return += [f'NDCG@{k}']
            except ListTooShortException as e:
                print(e)
                continue
        return full_df[columns_to_return].mean()
