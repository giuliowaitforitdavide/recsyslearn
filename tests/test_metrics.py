import unittest
import pandas as pd
import numpy as np
from recsyslearn.accuracy.metrics import NDCG
from recsyslearn.fairness.metrics import KullbackLeibler, Entropy, MutualInformation
from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from recsyslearn.errors.errors import FlagNotValidException
from tests.utils import first_example, item_groups, item_pop_perc, pos_items, rel_matrix_1, second_example, top_n_1, user_groups


class EntropyTest(unittest.TestCase):
    """
    Tester for the Entropy class.
    """

    def test_user_exposure(self) -> None:
        top_n = first_example.merge(user_groups, on='user')
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 0.91830, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 1.48547, delta=1e-5)


class KullbackLeiblerTest(unittest.TestCase):
    """
    Tester for the KullbackLeibler class.
    """

    def test_user_effectiveness(self) -> None:
        top_n = second_example.merge(user_groups, on='user')
        rel_matrix = rel_matrix_1.merge(user_groups, on='user')
        target_representation = pd.DataFrame([['1', 0.5], ['2', 0.5]], columns=['group', 'target_representation'])
        divergence = KullbackLeibler().evaluate(top_n, target_representation, rel_matrix)
        self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        target_representation = pd.DataFrame([['1', 0.2], ['2', 0.3], ['3', 0.5]],
                                             columns=['group', 'target_representation'])
        divergence = KullbackLeibler().evaluate(top_n, target_representation)
        self.assertAlmostEqual(divergence, 0, delta=1e-5)


class MutualInformationTest(unittest.TestCase):
    """
    Tester for the KullbackLeibler class.
    """

    def test_user_exposure(self) -> None:
        top_n = first_example.merge(user_groups, on='user')
        mi = MutualInformation().evaluate(top_n, 'user')
        self.assertAlmostEqual(mi, 0.25582, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        mi = MutualInformation().evaluate(top_n, 'item')
        self.assertAlmostEqual(mi, 0.10570, delta=1e-5)

    def test_flag_not_valid(self) -> None:
        with self.assertRaises(FlagNotValidException) as context:
            top_n = first_example.merge(item_groups, on='item')
            MutualInformation().evaluate(top_n, 'ratings')

        self.assertTrue('Invalid flag.' in str(context.exception))


class CoverageTest(unittest.TestCase):
    """
    Tester for the Coverage class.
    """

    def test_coverage_one(self) -> None:
        top_n = first_example[
            (first_example['item'] != '3') & (first_example['item'] != '4') & (first_example['item'] != '1')]
        cov = Coverage().evaluate(top_n, item_groups.item.tolist())
        self.assertAlmostEqual(cov, 0.7)

    def test_coverage_two(self) -> None:
        top_n = first_example[
            (first_example['item'] != '1') & (first_example['item'] != '2') & (first_example['item'] != '4') & (
                first_example['item'] != '7') & (first_example['item'] != '9') & (first_example['item'] != '10')]
        cov = Coverage().evaluate(top_n, item_groups.item.tolist())
        self.assertAlmostEqual(cov, 0.4)


class NoveltyTest(unittest.TestCase):
    """
    Tester for the Novelty class.
    """

    def setUp(self):
        novelty = np.vectorize(lambda x: -np.log2(x))
        self.novelty = lambda list: float(np.mean(novelty(list)))

    def test_novelty_one(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self.novelty(top_n['group'].to_numpy()), delta=1e-5)

    def test_novelty_two(self) -> None:
        top_n = second_example.merge(item_groups, on='item')
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self.novelty(top_n['group'].to_numpy()), delta=1e-5)

    def test_novelty_three(self) -> None:
        top_n = first_example.merge(item_pop_perc, on='item')
        nov = Novelty().evaluate(top_n, popularity_definition='percentage')
        self.assertAlmostEqual(nov, self.novelty(top_n['percentage'].to_numpy()), delta=1e-5)


class NDCGTest(unittest.TestCase):
    """
    Tester for the NDCG class.
    """

    def test_ndcg_one(self) -> None:
        ndcg_df = NDCG().evaluate(top_n_1, pos_items, ats=(2,))
        ndcg_vals = pd.DataFrame.from_dict(
            {
                '1': (1 / np.log(1 + 1) + 0 / np.log(2 + 1)) / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),  # OK
                '2': (0 / np.log(1 + 1) + 0 / np.log(2 + 1)) / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),  # OK
                '3': (0 / np.log(1 + 1) + 1 / np.log(2 + 1)) / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                '4': (0 / np.log(1 + 1) + 1 / np.log(2 + 1)) / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                '5': (1 / np.log(1 + 1) + 1 / np.log(2 + 1)) / (1 / np.log(1 + 1) + 1 / np.log(2 + 1)),
                '6': (1 / np.log(1 + 1) + 0 / np.log(2 + 1)) / (1 / np.log(1 + 1) + 0 / np.log(2 + 1))   # OK
             }, orient='index'
        )

        ndcg = ndcg_vals.mean()
        self.assertAlmostEqual(ndcg.iloc[0], ndcg_df.iloc[0], delta=1e-5)


if __name__ == '__main__':
    unittest.main()
