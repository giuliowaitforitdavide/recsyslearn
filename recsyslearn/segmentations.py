import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from recsyslearn.errors import SegmentationNotSupportedException, WrongProportionsException


class Segmentation(ABC):

	'''
	Abstract class for different group segmentations.
	'''

	def __init__(self) -> None:
		super().__init__()

	
	@abstractmethod
	def segment(self, dataset: pd.DataFrame) -> None:
		pass


class InteractionSegmentation(Segmentation):

	'''
	Segmentation of items based on the number of interaction they have.
	'''

	def segment(self, dataset: pd.DataFrame, proportion: list = [0.8, 0.2], min_interaction: int = 0) -> pd.DataFrame:

		'''
		Segmentation of items based on their interactions with different users.


		Parameters
		----------
		dataset : pd.DataFrame
			The complete dataset.

		proportion : list, default [0.8, 0.2]
			The proportion of interactions wanted for every group. Its length should be between 1 and 3.


		Raises
		------
		SegmentationNotSupportedException
			If len(proportion) not in (1, 2, 3).
		
		WrongProportionsException
			If sum(proportion) is not 1, which means it doesn't cover all the items/users.

		
		Return
		------
		DataFrame with items and belonging group.
		'''

		if len(proportion) not in (1, 2, 3):
			raise SegmentationNotSupportedException()

		if sum(proportion) != 1:
			raise WrongProportionsException()

		if len(proportion) == 1:
			return dataset

		item_groups = dataset.groupby('item').size().reset_index(name='count').sort_values('count', ascending=False) 

		tmp = item_groups[item_groups['count'] > min_interaction]
		n_int = tmp['count'].sum()

		short_thr = np.rint(proportion[0] * n_int)
		mid_thr = np.rint(proportion[1] * n_int) + short_thr
		tmp.loc[:, 'cumulative_sum'] = tmp['count'].cumsum()
		
		short_head = tmp.loc[tmp['cumulative_sum'].lt(short_thr), 'item']
		mid_tail = tmp.loc[tmp['cumulative_sum'].lt(mid_thr) & ~tmp['item'].isin(short_head), 'item']
		conditions = [item_groups['item'].isin(short_head), item_groups['item'].isin(mid_tail)]
		choices = ['1', '2']
		item_groups.loc[:, 'group'] = np.select(conditions, choices, default='3' if len(proportion) == 3 else '2')
		return item_groups[['item', 'group']]


class ActivitySegmentation(Segmentation):

	'''
	Segmentation of users based on their number of interaction.
	'''

	def segment(self, dataset: pd.DataFrame, proportion: list = [0.1, 0.9], min_interaction: int = 0) -> pd.DataFrame:

		'''
		Segmentation of users based on their interactions with different items.


		Parameters
		----------
		dataset : pd.DataFrame
			The complete dataset.

		proportion : list, default [0.8, 0.2]
			The proportion of interactions wanted for every group.


		Raises
		------
		SegmentationNotSupportedException
			If len(proportion) not in (1, 2, 3).
		
		WrongProportionsException
			If sum(proportion) is not 1, which means it doesn't cover all the items/users.

		
		Return
		------
		DataFrame with users and belonging group.
		'''

		if len(proportion) not in (1, 2, 3):
			raise SegmentationNotSupportedException()

		if sum(proportion) != 1:
			raise WrongProportionsException()

		if len(proportion) == 1:
			return dataset

		user_groups = dataset.groupby('user').size().reset_index(name='count')
		user_groups = user_groups[user_groups['count'] > min_interaction]
		
		# Adding some random interactions to remove ties
		user_groups.loc[:, 'count'] = user_groups.loc[:, 'count'].apply(lambda x: x + np.random.choice(list(range(10))))
		user_groups = user_groups.sort_values('count', ascending=False)
		user_groups.loc[:, 'count'] = np.arange(user_groups.shape[0]) + 1
		threshold = round(user_groups.shape[0] * proportion[0]) if round(user_groups.shape[0] * proportion[0]) > 0 else 1

		user_groups.loc[user_groups['count'] <= threshold, 'group'] = '1'
		user_groups.loc[user_groups['count'] > threshold, 'group'] = '2'

		return user_groups[['user', 'group']]
