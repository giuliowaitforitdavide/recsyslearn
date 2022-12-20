import unittest
import numpy as np
from recsyslearn.beyond_accuracy.metrics import Coverage, Novelty
from tests.utils import first_example, item_groups, item_pop_perc, second_example

class CoverageTest(unittest.TestCase):

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

    def setUp(self):
        novelty = np.vectorize(lambda x: -np.log2(x))
        self.novelty = lambda list: float(np.mean(novelty(list)))

    def test_novelty_one(self) -> None:
        top_n = first_example.merge(item_groups, on='item')
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self.novelty(
            top_n['group'].to_numpy()), delta=1e-5)

    def test_novelty_two(self) -> None:
        top_n = second_example.merge(item_groups, on='item')
        nov = Novelty().evaluate(top_n)
        self.assertAlmostEqual(nov, self.novelty(
            top_n['group'].to_numpy()), delta=1e-5)

    def test_novelty_three(self) -> None:
        top_n = first_example.merge(item_pop_perc, on='item')
        nov = Novelty().evaluate(top_n, popularity_definition='percentage')
        self.assertAlmostEqual(nov, self.novelty(
            top_n['percentage'].to_numpy()), delta=1e-5)


if __name__ == '__main__':
    unittest.main()
