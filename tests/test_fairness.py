import unittest
import pandas as pd
from recsyslearn.fairness.metrics import KullbackLeibler, Entropy, MutualInformation
from tests.utils import first_example, item_groups, rel_matrix_1, second_example, user_groups


class EntropyTest(unittest.TestCase):

    def test_user_exposure(self) -> None:
        top_n = first_example.merge(user_groups, on='user')
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 0.91830, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        entropy = Entropy().evaluate(top_n)
        self.assertAlmostEqual(entropy, 1.48547, delta=1e-5)


class KullbackLeiblerTest(unittest.TestCase):

    def test_user_effectiveness(self) -> None:
        top_n = second_example.merge(user_groups, on='user')
        rel_matrix = rel_matrix_1.merge(user_groups, on='user')
        target_representation = pd.DataFrame([['1', 0.5], ['2', 0.5]], columns=[
                                             'group', 'target_representation'])
        divergence = KullbackLeibler().evaluate(
            top_n, target_representation, rel_matrix)
        self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        target_representation = pd.DataFrame([['1', 0.2], ['2', 0.3], ['3', 0.5]],
                                             columns=['group', 'target_representation'])
        divergence = KullbackLeibler().evaluate(top_n, target_representation)
        self.assertAlmostEqual(divergence, 0, delta=1e-5)


class MutualInformationTest(unittest.TestCase):

    def test_user_exposure(self) -> None:
        top_n = first_example.merge(user_groups, on='user')
        mi = MutualInformation().evaluate(top_n, 'user')
        self.assertAlmostEqual(mi, 0.25582, delta=1e-5)

    def test_item_exposure(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        mi = MutualInformation().evaluate(top_n, 'item')
        self.assertAlmostEqual(mi, 0.10570, delta=1e-5)

    def test_flag_not_valid(self) -> None:
        with self.assertRaises(KeyError) as context:
            top_n = first_example.merge(item_groups, on='item')
            MutualInformation().evaluate(top_n, 'ratings')

        self.assertTrue('[None] not in index' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
