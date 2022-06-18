import unittest
import pandas as pd
import numpy as np
from recsyslearn.metrics import Entropy, KullbackLeibler, MutualInformation, Novelty, Coverage

user_groups = pd.DataFrame([
  ['1', '1'],
  ['2', '1'],
  ['3', '2'],
  ['4', '2'],
  ['5', '2'],
  ['6', '2']
  ], columns=['user', 'group'])

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

rel_matrix_2 = pd.DataFrame([
  ['1', '2', 1],
  ['1', '9', 1],
  ['2', '6', 1],
  ['2', '7', 1],
  ['3', '1', 1],
  ['3', '3', 1],
  ['3', '4', 1],
  ['3', '5', 1],
  ['3', '7', 1],
  ['3', '9', 1],
  ['4', '2', 1],
  ['4', '3', 1],
  ['4', '5', 1],
  ['4', '7', 1],
  ['4', '8', 1],
  ['4', '9', 1],
  ['5', '1', 1],
  ['5', '2', 1],
  ['5', '4', 1],
  ['5', '5', 1],
  ['5', '6', 1],
  ['5', '9', 1],
  ['6', '1', 1],
  ['6', '2', 1],
  ['6', '6', 1],
  ['6', '8', 1],
  ['6', '9', 1],
], columns=['user', 'item', 'rank'])

entropyEvaluator = Entropy()
klEvaluator = KullbackLeibler()
miEvaluator = MutualInformation()
coverageEvaluator = Coverage()
noveltyEvaluator = Novelty()

class EntropyTest(unittest.TestCase):

  '''
  Tester for the Entropy class.
  '''

  def test_user_exposure(self) -> None:
    top_n = first_example.merge(user_groups, on='user')
    entropy = entropyEvaluator.evaluate(top_n)
    self.assertAlmostEqual(entropy, 0.91830, delta=1e-5)
    print('')
    print(f'User exposure test passed with an entropy equal to {entropy}')
  

  def test_item_exposure(self) -> None:
    top_n = first_example.merge(item_groups, on='item')
    entropy = entropyEvaluator.evaluate(top_n)
    self.assertAlmostEqual(entropy, 1.48547, delta=1e-5)
    print('')
    print(f'Item exposure test passed with an entropy equal to {entropy}')


class KullbackLeiblerTest(unittest.TestCase):

  '''
  Tester for the KullbackLeibler class.
  '''

  def test_user_effectiveness(self) -> None:
    top_n = second_example.merge(user_groups, on='user')
    rel_matrix = rel_matrix_1.merge(user_groups, on='user')
    target_representation = pd.DataFrame([['1', 0.5], ['2', 0.5]], columns=['group', 'target_representation'])
    divergence = klEvaluator.evaluate(top_n, target_representation, rel_matrix)
    self.assertAlmostEqual(divergence, 0.08530, delta=1e-5)
    print('')
    print(f'User effectiveness test passed with a divergence equal to {divergence}')


  def test_item_exposure(self) -> None:
    top_n = first_example.merge(item_groups, on='item')
    target_representation = pd.DataFrame([['1', 0.2], ['2', 0.3], ['3', 0.5]], columns=['group', 'target_representation'])
    divergence = klEvaluator.evaluate(top_n, target_representation)
    self.assertAlmostEqual(divergence, 0, delta=1e-5)
    print('')
    print(f'Item exposure test passed with a divergence equal to {divergence}')


class MutualInformationTest(unittest.TestCase):

  '''
  Tester for the KullbackLeibler class.
  '''

  def test_user_exposure(self) -> None:
    top_n = first_example.merge(user_groups, on='user')
    mi = miEvaluator.evaluate(top_n, 'user')
    self.assertAlmostEqual(mi, 0.25582, delta=1e-5)
    print('')
    print(f'User exposure test passed with a mutual information equal to {mi}') 


  def test_item_exposure(self) -> None:
    top_n = first_example.merge(item_groups, on='item')
    mi = miEvaluator.evaluate(top_n, 'item')
    self.assertAlmostEqual(mi, 0.10570, delta=1e-5)
    print('')
    print(f'Item exposure test passed with a mutual information equal to {mi}') 


class CoverageTest(unittest.TestCase):

  def test_coverage_one(self) -> None:
    top_n = first_example[(first_example['item'] != '3') & (first_example['item'] != '4') & (first_example['item'] != '1')]
    cov = coverageEvaluator.evaluate(top_n, item_groups.item.tolist())
    self.assertAlmostEqual(cov, 0.7)
    print('')
    print(f'Coverage test one passed with a coverage equal to {cov}')


  def test_coverage_two(self) -> None:
    top_n = first_example[(first_example['item'] != '1') & (first_example['item'] != '2') & (first_example['item'] != '4') & (first_example['item'] != '7') & (first_example['item'] != '9') & (first_example['item'] != '10')]
    cov = coverageEvaluator.evaluate(top_n, item_groups.item.tolist())
    self.assertAlmostEqual(cov, 0.4)
    print('')
    print(f'Coverage test one passed with a coverage equal to {cov}')


class NoveltyTest(unittest.TestCase):

  def novelty(self, list: list) -> float:
    novelty = np.vectorize(lambda x: -np.log2(1 / int(x)))
    return np.sum(novelty(list))/len(list)

  
  def test_novelty_one(self) -> None:
    top_n = first_example.merge(item_groups, on='item')
    nov = noveltyEvaluator.evaluate(top_n)
    self.assertAlmostEqual(nov, self.novelty(top_n['group'].to_numpy()), delta=1e-5)
    print('')
    print(f'Novelty test passed with a novelty equal to {nov}') 


  def test_novelty_two(self) -> None:
    top_n = second_example.merge(item_groups, on='item')
    nov = noveltyEvaluator.evaluate(top_n)
    self.assertAlmostEqual(nov, self.novelty(top_n['group'].to_numpy()), delta=1e-5)
    print('')
    print(f'Novelty test passed with a novelty equal to {nov}') 


if __name__ == '__main__':
  unittest.main()
