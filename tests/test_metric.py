import unittest
import pandas as pd
import numpy as np
from recsyslearn.metrics import Entropy, KullbackLeibler, MutualInformation, Novelty, Coverage
from tests.utils import first_example, second_example, item_groups, user_groups, rel_matrix_1, rel_matrix_2

class EntropyTest(unittest.TestCase):

	'''
	Tester for the Entropy class.
	'''

	def setUp(self) -> None:
		self.evaluator = Entropy()


	def test_user_exposure(self) -> None:
		top_n = first_example.merge(user_groups, on='user')
		entropy = self.evaluator.evaluate(top_n)
		self.assertAlmostEqual(entropy, 0.91830, delta=1e-5)


	def test_item_exposure(self) -> None:
		top_n = first_example.merge(item_groups, on='item')
		entropy = self.evaluator.evaluate(top_n)
		self.assertAlmostEqual(entropy, 1.48547, delta=1e-5)


class KullbackLeiblerTest(unittest.TestCase):

	'''
	Tester for the KullbackLeibler class.
	'''

	def setUp(self) -> None:
		self.evaluator = KullbackLeibler()


	def test_user_effectiveness(self) -> None:
		top_n = second_example.merge(user_groups, on='user')
		rel_matrix = rel_matrix_1.merge(user_groups, on='user')
		target_representation = pd.DataFrame([['1', 0.5], ['2', 0.5]], columns=['group', 'target_representation'])
		divergence = self.evaluator.evaluate(top_n, target_representation, rel_matrix)
		self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)


	def test_item_exposure(self) -> None:
		top_n = first_example.merge(item_groups, on='item')
		target_representation = pd.DataFrame([['1', 0.2], ['2', 0.3], ['3', 0.5]], columns=['group', 'target_representation'])
		divergence = self.evaluator.evaluate(top_n, target_representation)
		self.assertAlmostEqual(divergence, 0, delta=1e-5)


class MutualInformationTest(unittest.TestCase):

	'''
	Tester for the KullbackLeibler class.
	'''

	def setUp(self) -> None:
		self.evaluator = MutualInformation()


	def test_user_exposure(self) -> None:
		top_n = first_example.merge(user_groups, on='user')
		mi = self.evaluator.evaluate(top_n, 'user')
		self.assertAlmostEqual(mi, 0.25582, delta=1e-5)


	def test_item_exposure(self) -> None:
		top_n = first_example.merge(item_groups, on='item')
		mi = self.evaluator.evaluate(top_n, 'item')
		self.assertAlmostEqual(mi, 0.10570, delta=1e-5)


class CoverageTest(unittest.TestCase):

	'''
	Tester for the Coverage class.
	'''

	def setUp(self) -> None:
		self.evaluator = Coverage()


	def test_coverage_one(self) -> None:
		top_n = first_example[(first_example['item'] != '3') & (first_example['item'] != '4') & (first_example['item'] != '1')]
		cov = self.evaluator.evaluate(top_n, item_groups.item.tolist())
		self.assertAlmostEqual(cov, 0.7)


	def test_coverage_two(self) -> None:
		top_n = first_example[(first_example['item'] != '1') & (first_example['item'] != '2') & (first_example['item'] != '4') & (first_example['item'] != '7') & (first_example['item'] != '9') & (first_example['item'] != '10')]
		cov = self.evaluator.evaluate(top_n, item_groups.item.tolist())
		self.assertAlmostEqual(cov, 0.4)


class NoveltyTest(unittest.TestCase):

	'''
	Tester for the Novelty class.
	'''

	def setUp(self):
		self.evaluator = Novelty()
		novelty = np.vectorize(lambda x: np.mean(-np.log2(int(x))))
		self.novelty = lambda list: np.mean(novelty(list))


	def test_novelty_one(self) -> None:
		top_n = first_example.merge(item_groups, on='item')
		nov = self.evaluator.evaluate(top_n)
		self.assertAlmostEqual(nov, self.novelty(top_n['group'].to_numpy()), delta=1e-5)


	def test_novelty_two(self) -> None:
		top_n = second_example.merge(item_groups, on='item')
		nov = self.evaluator.evaluate(top_n)
		self.assertAlmostEqual(nov, self.novelty(top_n['group'].to_numpy()), delta=1e-5)


if __name__ == '__main__':
	unittest.main()
