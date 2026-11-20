# Examples

Each script runs from the root of the repository (`uv run python examples/02_accuracy.py`)
and prints what it computes. The test-suite executes all of them, and the documentation
shows them in `docs/user/examples.rst`: a new script is added there too.

| Script | What it shows |
| --- | --- |
| [`01_input_tables.py`](01_input_tables.py) | The tables on hand-made data: lists ordered by `rank`, by `score` or by the rows, identifiers of any type, graded relevance in `rating`. |
| [`02_accuracy.py`](02_accuracy.py) | nDCG@k: cutoffs, a score per user, a bootstrap interval, who is evaluated, linear and exponential gain. |
| [`03_beyond_accuracy.py`](03_beyond_accuracy.py) | Catalog coverage with different catalogs; novelty, per user and for items without interactions. |
| [`04_segmentations.py`](04_segmentations.py) | Head and tail by share of interactions, groups by share of the population, popularity as a number, groups from an attribute. |
| [`05_fairness.py`](05_fairness.py) | Distribution of exposure and of effectiveness over groups of items and of users, entropy, divergence from several targets, mutual information. |
| [`06_scores_and_ties.py`](06_scores_and_ties.py) | A popularity baseline given with scores: expected values with tied scores, imposing an order, why coverage is refused. |
| [`07_compare_recommenders.py`](07_compare_recommenders.py) | A model, a popularity baseline and a random one on every metric, in one table. |
| [`08_errors.py`](08_errors.py) | What is refused and with which error. |

The data in [`__assets__`](__assets__) are implicit feedback: `train_dataset.csv` and
`test_dataset.csv` have the columns `user, item`, and `top_k.csv` holds the top-10 lists
of a model as `user, item, rank`.
