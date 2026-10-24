"""Similarity engine: weighted block cosine, exact or HNSW-backed retrieval, variant-aware ranking."""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .data import TYPE_WORDS, WEARER_WORDS, Catalogue, load_catalogue, name_tags
from .features import BLOCKS, build_blocks, combine

MODEL_FILE = Path(__file__).resolve().parent.parent / "config" / "model.json"

# Hand-set prior (D9): the fallback, and the baseline the learned weights must beat (D18)
PRIOR_WEIGHTS = {"text": 0.35, "image": 0.15, "brand": 0.15, "price": 0.10, "list_price": 0.05,
                 "colour": 0.08, "rating": 0.06, "weight": 0.06}
CACHE_SIZE = 4096  # (product, k) results kept per process (D20)
DENSE_BLOCKS = ("image",)  # stored as a dense array: 64 values per product, a sparse format only adds overhead


def sparse_scores(rows: sparse.csr_matrix, rows_t: sparse.csr_matrix, q: int) -> np.ndarray:
    """rows @ rows[q], reading only the columns product q actually uses (its row is very sparse) (D15)."""
    r = rows[q]
    return rows_t[r.indices].T @ r.data


def load_model(path: Path = MODEL_FILE) -> dict:
    """config/model.json (written by tune.py); the prior if it is missing."""
    return json.loads(path.read_text()) if path.exists() else {"weights": dict(PRIOR_WEIGHTS)}


def load_weights(path: Path = MODEL_FILE) -> Dict[str, float]:
    """Learned weights from config/model.json, else the prior."""
    return load_model(path)["weights"]


def product_groups(cat: Catalogue, image: np.ndarray) -> np.ndarray:
    """One id per real-world product (D13): listings that share a name, a product family or an identical
    photo are linked, and linked listings form one group. Only one listing per group is ever shown."""
    keys: Dict[tuple, int] = {}
    rows, cols = [], []
    for i in range(len(cat)):
        links = [("name", cat.name_keys[i]), ("family", cat.family_keys[i])]
        if image[i].any():   # no photo -> no image link
            links.append(("image", image[i].tobytes()))
        for key in links:
            rows.append(i)
            cols.append(keys.setdefault(key, len(keys)))
    links = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cat), len(keys)))
    graph = sparse.bmat([[None, links], [links.T, None]])   # products <-> keys, bipartite
    return connected_components(graph, directed=False)[1][:len(cat)]


def word_masks(names: List[str], vocab: Dict[str, str]) -> np.ndarray:
    """One bit per group in vocab, per product: which wearers (or product types) its name mentions."""
    bit = {g: 1 << i for i, g in enumerate(sorted(set(vocab.values())))}
    return np.array([sum(bit[t] for t in name_tags(n, vocab)) for n in names], dtype=np.int64)


def word_clash(masks: np.ndarray, q: int) -> np.ndarray:
    """True where both names mention groups and they share none (a "men's" query against a "women's" product)."""
    return (masks[q] != 0) & (masks != 0) & ((masks & masks[q]) == 0)


class SimilarityEngine:
    def __init__(self, cat: Catalogue, weights: Optional[Dict[str, float]] = None, search: str = "exact",
                 blocks: Optional[Dict] = None, category_penalty: float = 0.0, name_penalty: float = 0.0):
        self.cat = cat
        self.index = {pid: i for i, pid in enumerate(cat.ids)}
        self.ratings = np.array([r or 0.0 for r in cat.ratings])
        # integer rank of each id: final tie-breaker without sorting strings per query (D15)
        self.id_rank = np.argsort(np.argsort(np.array(cat.ids)))
        self.blocks = blocks if blocks is not None else build_blocks(cat)  # reusable across experiments
        self.weights = weights or load_weights()
        # Scoring layout (D6, D15): Σ w·cos = sparse part + dense image part
        scale = {b: np.float32(np.sqrt(w)) for b, w in zip(BLOCKS, validate_weights(self.weights))}
        self.blocks = {b: sparse.csr_matrix(m, dtype=np.float32) for b, m in self.blocks.items()}
        self._sparse = sparse.hstack([self.blocks[b] * scale[b] for b in BLOCKS if b not in DENSE_BLOCKS]).tocsr()
        self._sparse_t = self._sparse.T.tocsr()
        self._dense = np.hstack([self.blocks[b].toarray() * scale[b] for b in DENSE_BLOCKS])
        # The same layout per block, unweighted, for weight tuning and explanations
        self._block_t = {b: self.blocks[b].T.tocsr() for b in BLOCKS if b not in DENSE_BLOCKS}
        self._block_dense = {b: self.blocks[b].toarray() for b in DENSE_BLOCKS}
        # Which blocks hold data for each product, shape (n_blocks, n): drives the no-signal check (D23)
        self.has_data = np.vstack([np.diff(self.blocks[b].indptr) > 0 if b not in DENSE_BLOCKS
                                   else self._block_dense[b].any(axis=1) for b in BLOCKS])
        self.groups = product_groups(cat, self._block_dense["image"])
        # Substitute policy (D24): a known leaf category that differs from the query's known one costs this much
        self.category_penalty = float(category_penalty)
        codes = {c: i for i, c in enumerate(sorted({c for c in cat.categories if c}))}
        self.category_code = np.array([codes[c] if c else -1 for c in cat.categories])
        self._category_known = self.category_code >= 0
        # Where a category is missing, the wearer and product-type words in the names stand in for it (D28)
        self.name_penalty = float(name_penalty)
        self._wearer, self._kind = (word_masks(cat.names, vocab) for vocab in (WEARER_WORDS, TYPE_WORDS))
        self.ann = None
        if search == "hnsw":
            from .ann import HnswIndex
            self.matrix = combine(self.blocks, self.weights)   # one matrix: HNSW needs a single vector space
            self.ann = HnswIndex(self.matrix)
        # Per-engine result cache for the default weights; rebuilt with the engine when weights change
        self._cached = lru_cache(maxsize=CACHE_SIZE)(self._similar_ids)

    # ---------- scoring ----------

    def scores(self, q: int) -> np.ndarray:
        """Similarity of product q to every product: Σ_b w_b · cos_b(q, ·)."""
        return sparse_scores(self._sparse, self._sparse_t, q) + self._dense @ self._dense[q]

    def block_scores(self, q: int) -> np.ndarray:
        """Per-block cosine of q against every product, shape (n_blocks, n). Used for weight tuning."""
        return np.vstack([self._block_dense[b] @ self._block_dense[b][q] if b in DENSE_BLOCKS
                          else sparse_scores(self.blocks[b], self._block_t[b], q) for b in BLOCKS])

    def type_penalty(self, q: int, penalty: Optional[float] = None) -> np.ndarray:
        """Penalty per product for being a different kind of product from q.

        Both leaf categories known: `penalty` (the category penalty, D24) if they differ. Either category unknown
        (15% of products have none): the name penalty if the names mention wearers with nothing in common (men's vs
        women's) or product types with nothing in common (t-shirt vs jeans), D28. Otherwise 0.
        An explicit `penalty` of 0 (a request's `category_penalty=0`) turns both rules off.
        """
        off = penalty is not None and penalty == 0
        penalty = self.category_penalty if penalty is None else penalty
        code = self.category_code[q]
        both_known = self._category_known & (code >= 0)
        out = penalty * (both_known & (self.category_code != code)) if penalty else np.zeros(len(self.cat))
        if self.name_penalty and not off:
            clash = word_clash(self._wearer, q) | word_clash(self._kind, q)
            out = out + self.name_penalty * (clash & ~both_known)
        return out

    # ---------- ranking ----------

    def _ordered(self, scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Candidates best first; ties -> higher rating, then smaller id (D14)."""
        return candidates[np.lexsort((self.id_rank[candidates], -self.ratings[candidates], -scores[candidates]))]

    def _distinct(self, order: np.ndarray, q: int, k: int) -> List[int]:
        """Skip the query, duplicate listings and other listings of the same product (D13)."""
        seen_asins, seen_groups = {self.cat.asins[q]}, {self.groups[q]}
        picked = []
        for i in order:
            if self.cat.asins[i] in seen_asins or self.groups[i] in seen_groups:
                continue
            seen_asins.add(self.cat.asins[i])
            seen_groups.add(self.groups[i])
            picked.append(int(i))
            if len(picked) == k:
                break
        return picked

    def rank(self, scores: np.ndarray, q: int, k: int) -> List[int]:
        """Top-k distinct products from a full score vector, via partial sort (D15)."""
        n = len(scores)
        limit = min(4 * k + 10, n)   # headroom for skipped variants
        while True:
            if limit < n:
                threshold = np.partition(scores, -limit)[-limit]
                candidates = np.flatnonzero(scores >= threshold)  # keeps ties at the cut-off
            else:
                candidates = np.arange(n)
            picked = self._distinct(self._ordered(scores, candidates), q, k)
            if len(picked) == k or limit >= n:
                return picked
            limit *= 4

    def similar_indices(self, q: int, k: int) -> List[int]:
        if self.ann is None:
            return self.rank(self.scores(q) - self.type_penalty(q), q, k)
        # Two-stage (D19): HNSW proposes candidates, exact scores re-rank them
        from .ann import CANDIDATE_POOL
        candidates = self.ann.candidates(q, min(max(CANDIDATE_POOL, 4 * k + 10), len(self.cat)))
        exact = self.matrix[candidates] @ self.matrix[q].toarray().ravel() - self.type_penalty(q)[candidates]
        scores = np.full(len(self.cat), -1.0)
        scores[candidates] = exact
        picked = self._distinct(self._ordered(scores, candidates), q, k)
        # Rare: too many candidates were variants/duplicates -> fall back to exact search
        return picked if len(picked) == k else self.rank(self.scores(q) - self.type_penalty(q), q, k)

    def explain(self, q: int, results: List[int], weights: Optional[Dict[str, float]] = None,
                category_penalty: Optional[float] = None) -> List[Dict[str, float]]:
        """Per-block contribution (weight x cosine) of each result, plus the type penalty as a negative
        'category' entry; together they sum to the ranking score."""
        mix = validate_weights(weights or self.weights)
        cos = self.block_scores(q)[:, results]
        penalty = self.type_penalty(q, category_penalty)[results]
        return [{**{b: float(mix[j] * cos[j, r]) for j, b in enumerate(BLOCKS)}, "category": -float(penalty[r])}
                for r in range(len(results))]

    def signal(self, q: int, weights: Optional[Dict[str, float]] = None) -> Dict[str, list]:
        """Which weighted blocks this product has data for, and which are missing (shown in the demo)."""
        mix = validate_weights(weights or self.weights)
        used = [(j, b) for j, b in enumerate(BLOCKS) if mix[j] > 0]
        return {"available": [b for j, b in used if self.has_data[j, q]],
                "missing": [b for j, b in used if not self.has_data[j, q]]}

    def _similar_ids(self, q: int, k: int) -> tuple:
        return tuple(self.cat.ids[i] for i in self.similar_indices(q, k))

    def find_similar_products(self, product_id: str, num_similar: int,
                              weights: Optional[Dict[str, float]] = None,
                              category_penalty: Optional[float] = None) -> List[str]:
        """Default settings use the cached path; per-request `weights` re-mix the block scores (D20) and
        `category_penalty` overrides the substitute policy (D24)."""
        q = self.index[product_id]  # KeyError -> 404 in the API
        if num_similar < 1:
            raise ValueError("num_similar must be at least 1")
        if category_penalty is not None and not 0 <= category_penalty <= 1:
            raise ValueError("category_penalty must be between 0 and 1")
        if not weights and category_penalty is None:
            return list(self._cached(q, num_similar))
        mix = validate_weights(weights or self.weights)
        if not (mix * self.has_data[:, q]).any():
            # D23: every selected block is empty for this product, so every score would be 0 and the
            # "ranking" would be tie-breakers only. Refuse instead of returning confident-looking noise.
            raise ValueError(f"product {product_id} has no data for the selected block(s) "
                             f"{[b for j, b in enumerate(BLOCKS) if mix[j] > 0]}; it has data for "
                             f"{[b for j, b in enumerate(BLOCKS) if self.has_data[j, q]]}")
        scores = mix @ self.block_scores(q) - self.type_penalty(q, category_penalty)
        return [self.cat.ids[i] for i in self.rank(scores, q, num_similar)]


def group_by_name(engine: SimilarityEngine) -> SimilarityEngine:
    """Group listings by exact normalised name only, instead of name, family and photo. This is the grain of the
    co-purchase labels the weights are learned at (tune.py), and an ablation in the evaluations."""
    codes = {k: i for i, k in enumerate(sorted(set(engine.cat.name_keys)))}
    engine.groups = np.array([codes[k] for k in engine.cat.name_keys])
    return engine


def validate_weights(weights: Dict[str, float]) -> np.ndarray:
    """Per-request weights -> normalised vector over BLOCKS; ValueError (HTTP 422) if invalid."""
    if set(weights) - set(BLOCKS):
        raise ValueError(f"unknown block(s) {sorted(set(weights) - set(BLOCKS))}; choose from {BLOCKS}")
    try:
        mix = np.array([float(weights.get(b, 0.0)) for b in BLOCKS])
    except (TypeError, ValueError):
        raise ValueError("weights must be numbers") from None
    with np.errstate(over="ignore"):   # an overflowing sum is rejected just below
        total = mix.sum()
    if not np.all(np.isfinite(mix)) or (mix < 0).any() or not np.isfinite(total) or total <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    return mix / total


def parse_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """'image:0.6,text:0.4' -> {'image': 0.6, 'text': 0.4}; empty -> None (learned defaults)."""
    if not raw:
        return None
    try:
        pairs = [(name.strip(), float(value)) for name, value in (part.split(":") for part in raw.split(","))]
    except ValueError:
        raise ValueError("weights must look like 'text:0.5,image:0.5'") from None
    names = [name for name, _ in pairs]
    if len(set(names)) < len(names):
        raise ValueError("each block may appear only once in weights")
    return dict(pairs)


_engine: Optional[SimilarityEngine] = None


def get_engine() -> SimilarityEngine:
    """Process-wide engine; SEARCH_MODE=hnsw switches retrieval to the ANN index."""
    global _engine
    if _engine is None:
        policy = load_model().get("policy", {})
        _engine = SimilarityEngine(load_catalogue(), search=os.getenv("SEARCH_MODE", "exact"),
                                   category_penalty=policy.get("category_penalty", 0.0),
                                   name_penalty=policy.get("name_penalty", 0.0))
    return _engine


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    """The function signature required by the brief."""
    return get_engine().find_similar_products(product_id, num_similar)


def calculate_similarity(product_id: str, other_id: str) -> float:
    """The brief's calculate_similarity: the score find_similar_products ranks by, for one pair of products
    (sum of weight x cosine over the attribute blocks, minus the product-type penalty)."""
    engine = get_engine()
    return float(sum(engine.explain(engine.index[product_id], [engine.index[other_id]])[0].values()))
