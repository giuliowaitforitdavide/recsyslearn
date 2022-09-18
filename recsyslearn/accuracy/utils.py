import numpy as np


def ndcg(ranked_list, pos_items, relevance=None, at=None):

    """ Compute NDCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. """

    if relevance is None:
        relevance = np.ones_like(pos_items, dtype=np.int32)
    assert len(relevance) == pos_items.shape[0]

    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)
    ideal_dcg = dcg(np.sort(relevance)[::-1][:at])
    rank_dcg = dcg(rank_scores)
    if rank_dcg == 0:
        return 0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


def dcg(scores):

    """ Compute DCG score, based on https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi implementation. """

    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)
