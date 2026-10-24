"""Ranking invariants on the real catalogue: every property must hold for each of 100 random products."""
import random

import pytest

from simsearch.data import load_catalogue
from simsearch.engine import calculate_similarity, find_similar_products, get_engine

K = 10
PRODUCTS = random.Random(2026).sample(load_catalogue().ids, 100)


@pytest.mark.parametrize("pid", PRODUCTS)
def test_returns_k_distinct_other_products(pid):
    engine = get_engine()
    result = find_similar_products(pid, K)
    assert len(result) == K and len(set(result)) == K and pid not in result
    groups = [engine.groups[engine.index[r]] for r in result]
    assert len(set(groups)) == K and engine.groups[engine.index[pid]] not in groups   # one listing per product


@pytest.mark.parametrize("pid", PRODUCTS)
def test_calculate_similarity_never_rises_down_the_list(pid):
    scores = [calculate_similarity(pid, r) for r in find_similar_products(pid, K)]
    assert all(a >= b - 1e-9 for a, b in zip(scores, scores[1:]))


@pytest.mark.parametrize("pid", PRODUCTS)
def test_rankings_are_deterministic_and_prefix_consistent(pid):
    engine = get_engine()
    full = find_similar_products(pid, K)                              # default path, served from the cache
    fresh = engine.similar_indices(engine.index[pid], K)              # recomputed, bypassing the cache
    assert [engine.cat.ids[i] for i in fresh] == full
    assert find_similar_products(pid, 3) == full[:3]
