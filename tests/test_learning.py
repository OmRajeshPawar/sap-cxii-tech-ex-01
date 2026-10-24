"""Weight learning, uncertainty and splits (simsearch/evaluation.py, labels.py): the parts an ML reviewer probes."""
from pathlib import Path

import numpy as np

from conftest import product
from simsearch.data import build_catalogue
from simsearch.engine import PRIOR_WEIGHTS, SimilarityEngine
from simsearch.evaluation import candidate_weights, paired_bootstrap, tune_weights
from simsearch.features import BLOCKS, build_blocks
from simsearch.labels import build_labels, query_splits, train_test_split

NO_PHOTOS = Path("no/such/embeddings.npz")


def brand_world():
    """Six queries whose co-purchased partner shares only the brand, each with a decoy that shares only the text."""
    rows = []
    for i in range(6):
        rows += [product(f"q{i}", f"alpha{i} shirt", brand_=f"Brand{i}", also=f"partner{i} hat"),
                 product(f"p{i}", f"partner{i} hat", brand_=f"Brand{i}"),
                 product(f"d{i}", f"alpha{i} shirt blue", brand_=f"Decoy{i}")]
    return build_catalogue(rows)


def test_weight_search_recovers_a_planted_signal():
    cat = brand_world()
    labels = build_labels(cat)
    engine = SimilarityEngine(cat, weights=dict(PRIOR_WEIGHTS), blocks=build_blocks(cat, NO_PHOTOS))
    trials = candidate_weights(40, np.random.default_rng(0), PRIOR_WEIGHTS)
    weights, ndcg = tune_weights(engine, sorted(labels), labels, trials, n_refine=20)
    assert ndcg[2] < 1.0 and ndcg.max() == 1.0       # text-only falls for the decoy; the search finds weights that don't
    learned = SimilarityEngine(cat, weights=weights, blocks=engine.blocks)
    for q in labels:                                 # with the learned weights the co-purchased partner ranks first
        assert cat.name_keys[learned.similar_indices(q, 1)[0]] in labels[q]
    assert weights["brand"] > 0


def test_candidate_weights_are_on_the_simplex_and_keep_the_anchors_first():
    trials = candidate_weights(50, np.random.default_rng(0), PRIOR_WEIGHTS)
    assert trials.shape == (53, len(BLOCKS)) and np.allclose(trials.sum(axis=1), 1) and (trials >= 0).all()
    assert np.allclose(trials[0], np.array([PRIOR_WEIGHTS[b] for b in BLOCKS]) / sum(PRIOR_WEIGHTS.values()))
    assert np.allclose(trials[1], 1 / len(BLOCKS)) and trials[2][BLOCKS.index("text")] == 1


def test_paired_bootstrap_is_exact_on_easy_cases_and_deterministic():
    a = np.random.default_rng(1).random(500)
    same = paired_bootstrap(a, a)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    shifted = paired_bootstrap(a + 0.1, a)            # pairing removes all query-to-query noise
    assert np.isclose(shifted["delta"], 0.1) and np.allclose(shifted["ci95"], 0.1) and shifted["share_boot_le_0"] == 0
    noisy = paired_bootstrap(a, np.random.default_rng(2).random(500))
    assert noisy["ci95"][0] < noisy["delta"] < noisy["ci95"][1] and noisy == paired_bootstrap(a, np.random.default_rng(2).random(500))


def test_query_splits_are_disjoint_deterministic_and_respect_the_train_side():
    labels = {i: {f"x{i}"} for i in range(400)}
    tune, val, check = query_splits(labels, seed=0, n_tune=50, n_val=80, n_test=60)
    assert (len(tune), len(val), len(check)) == (50, 80, 60)
    assert not (set(tune) & set(val)) and not ((set(tune) | set(val)) & set(check))
    train, test = train_test_split(labels, seed=0)
    assert set(tune) | set(val) <= set(train) and set(check) <= set(test)
    assert (tune, val, check) == query_splits(labels, seed=0, n_tune=50, n_val=80, n_test=60)
