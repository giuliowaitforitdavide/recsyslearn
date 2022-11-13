import numpy as np
import pandas as pd
from abc import ABC
from recsyslearn.utils import check_columns_exist


class BeyondAccuracyMetric(ABC):

    """
    Abstract Class for Metrics.
    """

    def __init__(self) -> None:
        return


class Coverage(BeyondAccuracyMetric):

    """
    Coverage evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    """

    @classmethod
    def evaluate(cls, top_n: pd.DataFrame, items: list) -> float:
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
        ColumnsNotExistException
            If top_n not in the form ('user', 'item', 'rank', 'group').


        Return
        ------
        The computed coverage.
        """

        check_columns_exist(top_n, ['user', 'item', 'rank'])
        return len(top_n.item.unique().tolist()) / len(items)


class Novelty(BeyondAccuracyMetric):

    """
    Novelty evaluator for recommender systems.
    Used formula can be found here https://doi.org/10.1007/s13735-018-0154-2
    where popularity is defined in terms of the segmentation of the item groups
    (e.g. short head -> 3, mid tail   -> 2, long tail  -> 1)
    or in terms of percentage of user-item interactions.
    """

    @classmethod
    def evaluate(cls, top_n: pd.DataFrame, popularity_definition='group') -> float:
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
        ColumnsNotExistException
            If top_n not in the form ('user', 'item', 'rank', popularity_definition).


        Return
        ------
        The computed novelty.
        """

        check_columns_exist(
            top_n, ['user', 'item', 'rank', popularity_definition])
        top_n.loc[:, popularity_definition] = pd.to_numeric(
            top_n.loc[:, popularity_definition])
        top_n = top_n.groupby('user')[popularity_definition].apply(
            lambda x: np.mean(- np.log2(x)))
        return top_n.mean()
