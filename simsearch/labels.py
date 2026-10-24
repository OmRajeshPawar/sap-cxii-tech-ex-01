"""Weak relevance labels and offline ranking metrics (D16-D17).

The dataset has no "similar product" ground truth, but 24k rows carry
`other_items_customers_buy`: names of items shoppers bought alongside this one.
Where those names match products in the catalogue, they are behavioural evidence of
relatedness, so they serve as weak positives. They are never used as features (no leakage).
"""
import random
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .data import Catalogue


def build_labels(cat: Catalogue) -> Dict[int, Set[str]]:
    """Query index -> set of relevant name keys. One query per product name (variants share labels)."""
    known = set(cat.name_keys)
    labels, seen_names = {}, set()
    for q, also in enumerate(cat.also_bought):
        own = cat.name_keys[q]
        relevant = {k for k in also if k in known and k != own}  # a variant of itself is never returned
        if relevant and own not in seen_names:
            seen_names.add(own)
            labels[q] = relevant
    return labels


def train_test_split(labels: Dict[int, Set[str]], test_fraction: float = 0.5,
                     seed: int = 0) -> Tuple[List[int], List[int]]:
    """Random split of queries; one query per product name, so exact-name variants never sit on both sides."""
    queries = sorted(labels)
    random.Random(seed).shuffle(queries)
    cut = int(len(queries) * (1 - test_fraction))
    return queries[:cut], queries[cut:]


def query_splits(labels: Dict[int, Set[str]], seed: int = 0, n_tune: int = 1000, n_val: int = 2000,
                 n_test: int = 2000) -> Tuple[List[int], List[int], List[int]]:
    """The three query sets every script shares (D17).

    tune: learns block weights. val: every other choice (photo encoder, rejected experiments, type penalties),
    drawn from the train side but disjoint from tune. check: an early test sample that was looked at during
    development, so reported numbers use fresh_queries() instead (D25); it is kept to rebuild the exclusion set.
    """
    train, test = train_test_split(labels, seed=seed)
    perm = [int(q) for q in np.random.default_rng(seed).permutation(train)]
    sampled = np.random.default_rng(seed + 1).choice(test, size=min(n_test, len(test)), replace=False)
    test_sample = [int(q) for q in sampled]
    return perm[:n_tune], perm[n_tune:n_tune + n_val], test_sample


def ranking_metrics(retrieved: List[str], relevant: Set[str], k: int = 10) -> Dict[str, float]:
    """Recall@k, nDCG@k, HitRate@k and MRR@k for one query (binary relevance).

    A relevant key earns credit once: a second listing of the same relevant item is not a second hit.
    Recall divides by every relevant item (standard recall); nDCG's ideal ranking fills min(|relevant|, k) slots.
    """
    seen, hits = set(), []
    for key in retrieved[:k]:
        hits.append(1.0 if key in relevant and key not in seen else 0.0)
        seen.add(key)
    ideal = min(len(relevant), k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    first = next((i for i, h in enumerate(hits) if h), None)
    return {
        "recall": sum(hits) / len(relevant),
        "ndcg": float(np.dot(hits, discounts[:len(hits)]) / discounts[:ideal].sum()),
        "hit_rate": float(any(hits)),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
    }


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}


# ---------- evaluation protocol (D25): product families, family-disjoint fresh test queries ----------

def build_family_labels(cat: Catalogue) -> Dict[int, Set[str]]:
    """Query index -> relevant family keys. Labels work on product families because the service shows one
    listing per family (D13): another size of a co-purchased item is the same answer, and the query's own
    family is never returned, so it is never relevant. One query per family."""
    family_of = dict(zip(cat.name_keys, cat.family_keys))
    labels, seen = {}, set()
    for q, also in enumerate(cat.also_bought):
        own = cat.family_keys[q]
        relevant = {family_of[k] for k in also if k in family_of} - {own}
        if relevant and own not in seen:
            seen.add(own)
            labels[q] = relevant
    return labels


def fresh_queries(labels: Dict[int, Set[str]], groups: np.ndarray, used: Sequence[int]) -> List[int]:
    """Labelled queries whose product group shares nothing with any query used before (D25).

    `used` holds every query that has informed a choice or a reported number: tuning, validation and the
    early check sample. Group = same name, family or identical photo (engine.product_groups), so a size variant
    or a re-listed photo of a used query is excluded too, not only its exact name.
    """
    used_groups = {groups[q] for q in used}
    return sorted(q for q in labels if groups[q] not in used_groups)
