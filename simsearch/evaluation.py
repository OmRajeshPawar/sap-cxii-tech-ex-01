"""Offline evaluation and weight learning, shared by tune.py, evaluate.py and tools/."""
import random
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .engine import SimilarityEngine
from .features import BLOCKS
from .labels import mean_metrics, ranking_metrics

K = 10
REFINE_TRIALS, REFINE_CONCENTRATION = 200, 100.0   # stage 2 of the weight search (D18)


def evaluate(engine: SimilarityEngine, queries: Sequence[int], labels: Dict[int, Set[str]],
             k: int = K, per_query: bool = False, level: str = "name"):
    """Mean Recall/nDCG/HitRate/MRR@k plus sanity metrics, using the engine's real retrieval path.
    Same-category share counts only results where both leaf categories are known.

    level: the key the labels use, "name" (labels.build_labels, the grain the weights are tuned at) or "family"
    (labels.build_family_labels, the grain results are evaluated at). per_query=True also returns each query's nDCG, for paired_bootstrap.
    """
    keys = engine.cat.family_keys if level == "family" else engine.cat.name_keys
    cats = engine.cat.categories
    rows, same_cat, shown = [], [], set()
    for q in queries:
        idx = engine.similar_indices(q, k)
        rows.append(ranking_metrics([keys[i] for i in idx], labels[q], k))
        known = [cats[i] == cats[q] for i in idx if cats[i] and cats[q]]   # unknown category: no evidence either way
        if known:
            same_cat.append(np.mean(known))
        shown.update(idx)
    metrics = {**mean_metrics(rows), "same_category": float(np.mean(same_cat)) if same_cat else None,
               "coverage": len(shown) / len(engine.cat)}
    return (metrics, np.array([r["ndcg"] for r in rows])) if per_query else metrics


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean per-query difference a - b with a 95% bootstrap interval (queries resampled, pairs kept).

    Both systems are scored on the same queries, so pairing removes query difficulty from the noise.
    """
    diff = np.asarray(a) - np.asarray(b)
    means = diff[np.random.default_rng(seed).integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": float(diff.mean()), "ci95": [float(lo), float(hi)], "share_boot_le_0": float((means <= 0).mean())}


def same_category_baseline(cat, queries: Sequence[int], labels, k: int = K, seed: int = 0, level: str = "name"):
    """A stronger strawman: k random products from the query's own leaf category."""
    rng = random.Random(seed)
    by_cat: Dict[str, List[int]] = {}
    for i, c in enumerate(cat.categories):
        by_cat.setdefault(c, []).append(i)
    keys, rows = cat.family_keys if level == "family" else cat.name_keys, []
    for q in queries:
        pool = [i for i in by_cat[cat.categories[q]] if i != q]
        picks = rng.sample(pool, min(k, len(pool)))
        rows.append(ranking_metrics([keys[i] for i in picks], labels[q], k))
    return mean_metrics(rows)


def candidate_weights(n_trials: int, rng: np.random.Generator, prior: Dict[str, float]) -> np.ndarray:
    """Prior, equal and text-only anchors plus Dirichlet samples over the weight simplex."""
    anchors = np.array([[prior.get(b, 0.0) for b in BLOCKS],
                        [1.0] * len(BLOCKS),
                        [float(b == "text") for b in BLOCKS]])
    trials = np.vstack([anchors, rng.dirichlet(np.ones(len(BLOCKS)), size=n_trials)])
    return trials / trials.sum(axis=1, keepdims=True)


def score_trials(engine: SimilarityEngine, queries: Sequence[int], labels: Dict[int, Set[str]],
                 trials: np.ndarray, k: int = K) -> np.ndarray:
    """Mean nDCG@k of every weight vector. Per-block scores are computed once per query and shared."""
    totals = np.zeros(len(trials))
    keys = engine.cat.name_keys
    for q in queries:
        for t, scores in enumerate(trials @ engine.block_scores(q)):
            retrieved = [keys[i] for i in engine.rank(scores, q, k)]
            totals[t] += ranking_metrics(retrieved, labels[q], k)["ndcg"]
    return totals / len(queries)


def tune_weights(engine: SimilarityEngine, queries: Sequence[int], labels: Dict[int, Set[str]],
                 trials: np.ndarray, k: int = K, n_refine: int = REFINE_TRIALS,
                 seed: int = 0) -> Tuple[Dict[str, float], np.ndarray]:
    """Coarse-to-fine random search on the (tuning) queries.

    Stage 1 scores the given trials spread over the whole simplex. Stage 2 samples n_refine points
    from a Dirichlet centred on the stage-1 winner (concentration REFINE_CONCENTRATION, so each weight
    moves by a few hundredths) and keeps the best of both. Returns the weights and every trial's nDCG
    (stage 1 first, so the anchors stay at indices 0-2).
    """
    ndcg = score_trials(engine, queries, labels, trials, k)
    if n_refine:
        centre = trials[int(np.argmax(ndcg))]
        local = np.random.default_rng(seed).dirichlet(centre * REFINE_CONCENTRATION + 1e-3, size=n_refine)
        trials = np.vstack([trials, local])
        ndcg = np.concatenate([ndcg, score_trials(engine, queries, labels, local, k)])
    best = trials[int(np.argmax(ndcg))]
    return {b: round(float(w), 4) for b, w in zip(BLOCKS, best)}, ndcg
