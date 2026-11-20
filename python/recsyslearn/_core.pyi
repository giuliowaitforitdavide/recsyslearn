"""Type stubs of the native extension module."""

import numpy as np
from numpy.typing import NDArray

class RankedLists:
    def __new__(
        cls,
        users: NDArray[np.uint32],
        items: NDArray[np.uint32],
        ranks: NDArray[np.float64] | None,
        n_users: int,
    ) -> RankedLists: ...
    @staticmethod
    def from_scores(
        users: NDArray[np.uint32],
        items: NDArray[np.uint32],
        scores: NDArray[np.float64],
        n_users: int,
    ) -> RankedLists: ...

class GroundTruth:
    def __new__(
        cls,
        users: NDArray[np.uint32],
        items: NDArray[np.uint32],
        relevance: NDArray[np.float64] | None,
        n_users: int,
    ) -> GroundTruth: ...

def ndcg(
    lists: RankedLists, truth: GroundTruth, k: int | None, gain: str
) -> NDArray[np.float64]: ...
def mean_over_users(scores: NDArray[np.float64]) -> float | None: ...
def catalog_coverage(lists: RankedLists, k: int | None, n_items: int) -> float: ...
def novelty(
    lists: RankedLists, k: int | None, degrees: NDArray[np.uint64], n_users: int
) -> NDArray[np.float64]: ...
def benefit_distribution(
    lists: RankedLists,
    k: int | None,
    exposure: str,
    truth: GroundTruth | None,
    user_groups: NDArray[np.uint32] | None,
    n_user_groups: int,
    item_groups: NDArray[np.uint32] | None,
    n_item_groups: int,
    singletons: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]: ...
def entropy(distribution: NDArray[np.float64], normalize: bool) -> float: ...
def kl_divergence(
    distribution: NDArray[np.float64], target: NDArray[np.float64] | None
) -> float: ...
def interaction_counts(codes: NDArray[np.uint32], n: int) -> NDArray[np.uint64]: ...
def distinct_counts(
    entities: NDArray[np.uint32], counterparts: NDArray[np.uint32], n: int
) -> NDArray[np.uint64]: ...
def shares(counts: NDArray[np.uint64], total: int) -> NDArray[np.float64]: ...
def by_cumulative_interactions(
    counts: NDArray[np.uint64], proportions: list[float]
) -> NDArray[np.uint32]: ...
def by_interaction_count(
    counts: NDArray[np.uint64], proportions: list[float]
) -> NDArray[np.uint32]: ...
