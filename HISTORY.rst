=======
History
=======

0.4.0-alpha (2022-06-24)
------------------------

* First release on PyPI.

0.4.1 (2022-06-27)
-------------------

* Fixed novelty formula.

0.5.0 (2022-07-19)
-------------------

* Added a new item segmentation method, which gives a percentage score to the items based on their popularity.
* More accurate docs, with a beautiful theme.

0.5.1 (2022-07-27)
-------------------

* Fixed mantissa problem with the sum of proportion in the segmentation part.
* Improved code readability and tests coverage.

0.6.0 (2022-08-30)
-------------------

* Added accuracy computation with NDCG@k

1.0.0 (2022-12-21)
-------------------

* Added usage example
* Breaking library refactor
* Improved jobs' workflows, linting and other developing stuffs
* Fix typos in docs and functions

1.0.1 (2023-02-15)
-------------------

* Bump certifi from 2022.9.14 to 2022.12.7
* Added compatibility with pyenv for contributors
* Fixed segmentation based on interactions (now compatible with users)
* Fixed number of supported groups for users' activity (now could be three)
