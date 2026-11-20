"""Naive, loop-based transcriptions of the published formulas.

They are deliberately slow and independent from the library (no shared helper): the
test-suite checks the Rust implementation against them on random data.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from fractions import Fraction


def lists_from_rows(rows):
    """``{user: [items, best first]}`` from ``(user, item, rank)`` rows."""
    lists = defaultdict(list)
    for user, item, _rank in sorted(rows, key=lambda row: row[2]):
        lists[user].append(item)
    return dict(lists)


def ndcg(lists, relevance, k, exponential=False):
    """Mean nDCG@k over the users with at least one relevant item.

    ``relevance`` maps ``user -> {item: grade}``.
    """

    def gain(grade):
        return 2.0**grade - 1.0 if exponential else float(grade)

    scores = []
    for user, grades in relevance.items():
        ideal = sorted((gain(grade) for grade in grades.values()), reverse=True)[:k]
        idcg = sum(g / math.log2(position + 1) for position, g in enumerate(ideal, 1))
        if idcg == 0:
            continue
        dcg = sum(
            gain(grades.get(item, 0)) / math.log2(position + 1)
            for position, item in enumerate(lists.get(user, [])[:k], 1)
        )
        scores.append(dcg / idcg)
    return sum(scores) / len(scores)


def coverage(lists, catalog, k):
    recommended = {item for items in lists.values() for item in items[:k]}
    return len(recommended) / len(set(catalog))


def novelty(lists, interactions, k):
    """Mean over users of the mean ``log2(|U| / |U_i|)`` of their top-k items."""
    users_of = defaultdict(set)
    for user, item in interactions:
        users_of[item].add(user)
    n_users = len({user for user, _ in interactions})
    per_user = []
    for items in lists.values():
        top = items[:k]
        per_user.append(
            sum(math.log2(n_users / max(len(users_of[item]), 1)) for item in top) / len(top)
        )
    return sum(per_user) / len(per_user)


def joint_distribution(lists, k, block_of_user, block_of_item, flat=False, relevance=None):
    """``{(user block, item block): probability}`` of the normalised benefit."""
    mass = defaultdict(float)
    for user, items in lists.items():
        for position, item in enumerate(items[:k], 1):
            benefit = 1.0 if flat else 1.0 / math.log2(1 + position)
            if relevance is not None:
                benefit *= relevance.get(user, {}).get(item, 0.0)
            mass[block_of_user(user), block_of_item(item)] += benefit
    total = sum(mass.values())
    return {blocks: value / total for blocks, value in mass.items()}


def marginal(joint, axis):
    result = defaultdict(float)
    for blocks, probability in joint.items():
        result[blocks[axis]] += probability
    return dict(result)


def entropy(distribution):
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)


def kl_divergence(distribution, target):
    return sum(p * math.log2(p / target[group]) for group, p in distribution.items() if p > 0)


def mutual_information(joint):
    users, items = marginal(joint, 0), marginal(joint, 1)
    return sum(p * math.log2(p / (users[x] * items[y])) for (x, y), p in joint.items() if p > 0)


def _by_decreasing_count(counts):
    """Entities sorted by decreasing count, ties by ascending identifier."""
    return sorted(counts, key=lambda entity: (-counts[entity], entity))


def _cumulative(proportions):
    """Exact cumulative proportions, without the last one."""
    total, result = Fraction(0), []
    for proportion in proportions[:-1]:
        total += Fraction(str(proportion))
        result.append(total)
    return result


def by_cumulative_interactions(entities, proportions):
    """``{entity: group}``: group g (from 1) holds the entities reached before the
    cumulative interactions get to the g-th cumulative proportion of the total."""
    counts = Counter(entities)
    total = sum(counts.values())
    thresholds = [share * total for share in _cumulative(proportions)]
    groups, before = {}, 0
    for entity in _by_decreasing_count(counts):
        groups[entity] = 1 + sum(before >= threshold for threshold in thresholds)
        before += counts[entity]
    return groups


def by_interaction_count(entities, proportions):
    """``{entity: group}``: the first groups hold the top ``ceil(share * n)`` entities."""
    counts = Counter(entities)
    boundaries = [math.ceil(share * len(counts)) for share in _cumulative(proportions)]
    return {
        entity: 1 + sum(position >= boundary for boundary in boundaries)
        for position, entity in enumerate(_by_decreasing_count(counts))
    }


def popularity(pairs, relative_to_population):
    """``{entity: share}`` for the first element of each ``(entity, counterpart)`` pair."""
    if relative_to_population:
        pairs = set(pairs)
        total = len({counterpart for _, counterpart in pairs})
    else:
        total = len(pairs)
    return {entity: count / total for entity, count in Counter(e for e, _ in pairs).items()}
