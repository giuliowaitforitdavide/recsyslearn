import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from recsyslearn.fairness.metrics import KullbackLeibler, Entropy, MutualInformation

item_groups = pd.DataFrame([
    ['1', '1'],
    ['2', '1'],
    ['3', '2'],
    ['4', '2'],
    ['5', '2'],
    ['6', '3'],
    ['7', '3'],
    ['8', '3'],
    ['9', '3'],
    ['10', '3']
], columns=['item', 'group'])

user_groups = pd.DataFrame([
    ['1', '1'],
    ['2', '1'],
    ['3', '2'],
    ['4', '2'],
    ['5', '2'],
    ['6', '2']
], columns=['user', 'group'])

first_example = pd.DataFrame([
    ['1', '3', 1],
    ['1', '4', 1],
    ['1', '6', 1],
    ['1', '7', 1],
    ['1', '8', 1],
    ['2', '1', 1],
    ['2', '2', 1],
    ['2', '5', 1],
    ['2', '6', 1],
    ['2', '7', 1],
    ['3', '2', 1],
    ['3', '3', 1],
    ['3', '6', 1],
    ['3', '9', 1],
    ['3', '10', 1],
    ['4', '1', 1],
    ['4', '3', 1],
    ['4', '6', 1],
    ['4', '7', 1],
    ['4', '9', 1],
    ['5', '1', 1],
    ['5', '3', 1],
    ['5', '5', 1],
    ['5', '7', 1],
    ['5', '9', 1],
    ['6', '2', 1],
    ['6', '3', 1],
    ['6', '5', 1],
    ['6', '9', 1],
    ['6', '10', 1],
], columns=['user', 'item', 'rank'])

second_example = pd.DataFrame([
    ['1', '1', 1],
    ['1', '3', 6],
    ['1', '4', 3],
    ['1', '5', 4],
    ['1', '8', 5],
    ['1', '9', 2],
    ['2', '2', 6],
    ['2', '3', 5],
    ['2', '6', 4],
    ['2', '7', 3],
    ['2', '8', 2],
    ['2', '9', 1],
    ['3', '2', 1],
    ['3', '3', 2],
    ['3', '4', 3],
    ['3', '7', 4],
    ['3', '8', 5],
    ['3', '9', 6],
    ['4', '1', 4],
    ['4', '3', 5],
    ['4', '4', 6],
    ['4', '7', 3],
    ['4', '8', 2],
    ['4', '9', 1],
    ['5', '1', 6],
    ['5', '3', 2],
    ['5', '5', 3],
    ['5', '6', 4],
    ['5', '8', 5],
    ['5', '9', 1],
    ['6', '2', 3],
    ['6', '3', 5],
    ['6', '5', 4],
    ['6', '6', 6],
    ['6', '8', 2],
    ['6', '9', 1]
], columns=['user', 'item', 'rank'])

top_n_1 = pd.DataFrame([
    ['1', '2', 1],
    ['1', '8', 2],
    ['1', '5', 3],
    ['1', '3', 4],
    ['1', '6', 5],
    ['2', '1', 1],
    ['2', '2', 2],
    ['2', '6', 3],
    ['2', '7', 4],
    ['2', '9', 5],
    ['3', '6', 1],
    ['3', '5', 2],
    ['3', '8', 3],
    ['3', '7', 4],
    ['3', '9', 5],
    ['4', '1', 1],
    ['4', '2', 2],
    ['4', '3', 3],
    ['4', '4', 4],
    ['4', '5', 5],
    ['5', '5', 1],
    ['5', '4', 2],
    ['5', '3', 3],
    ['5', '2', 4],
    ['5', '1', 5],
    ['6', '9', 1],
    ['6', '8', 2],
    ['6', '6', 3],
    ['6', '3', 4],
    ['6', '4', 5],
], columns=['user', 'item', 'rank'])

rel_matrix_1 = pd.DataFrame([
    ['1', '2', 1],
    ['1', '9', 1],
    ['2', '6', 1],
    ['2', '7', 1],
    ['3', '1', 1],
    ['3', '7', 1],
    ['3', '9', 1],
    ['4', '3', 1],
    ['4', '7', 1],
    ['6', '2', 1],
    ['6', '9', 1]
], columns=['user', 'item', 'rank'])


class EntropyTest():

    def test_user_exposure(self) -> None:
        expected_entropy_df = pd.DataFrame([
            ['1', 0.528321],
            ['2', 0.389975]
        ], columns=['group', 'rank'])
        top_n = first_example.merge(user_groups, on='user')
        entropy_df = Entropy().evaluate(top_n)
        assert_frame_equal(
            entropy_df,
            expected_entropy_df,
            check_like=True,
            check_exact=False,
            rtol=1e-5
        )

    def test_item_exposure(self) -> None:
        expected_entropy_df = pd.DataFrame([
            ['1', 0.464386],
            ['2', 0.521090],
            ['3', 0.500000],
        ], columns=['group', 'rank'])
        top_n = first_example.merge(item_groups, on='item')
        entropy_df = Entropy().evaluate(top_n)
        assert_frame_equal(
            entropy_df,
            expected_entropy_df,
            check_like=True,
            check_exact=False,
            rtol=1e-5
        )


class KullbackLeiblerTest():

    def test_user_effectiveness(self) -> None:
        expected_divergence_df = pd.DataFrame([
            ['1', -0.198011],
            ['2', 0.283312]
        ], columns=['group', 'rank'])
        top_n = second_example.merge(user_groups, on='user')
        rel_matrix = rel_matrix_1.merge(user_groups, on='user')
        target_representation = pd.DataFrame([
            ['1', 0.5],
            ['2', 0.5]
        ], columns=['group', 'target_representation'])
        divergence_df = KullbackLeibler().evaluate(
            top_n, target_representation, rel_matrix)
        assert_frame_equal(
            divergence_df,
            expected_divergence_df,
            check_like=True,
            check_exact=False,
            rtol=1e-5
        )

    def test_item_exposure(self) -> None:
        expected_divergence_df = pd.DataFrame([
            ['1', 0.0],
            ['2', 0.0],
            ['3', 0.0],
        ], columns=['group', 'rank'])
        top_n = first_example.merge(item_groups, on='item')
        target_representation = pd.DataFrame([['1', 0.2], ['2', 0.3], ['3', 0.5]],
                                             columns=['group', 'target_representation'])
        divergence_df = KullbackLeibler().evaluate(top_n, target_representation)
        assert_frame_equal(
            divergence_df,
            expected_divergence_df,
            check_like=True,
            check_exact=False,
            rtol=1e-5
        )


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
        with self.assertRaises(KeyError):
            top_n = first_example.merge(item_groups, on='item')
            MutualInformation().evaluate(top_n, 'ratings')


if __name__ == '__main__':
    unittest.main()
