"""Feature blocks. Each block is L2-normalised per row, so block similarity = cosine (D2-D8, D10-D12)."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import Catalogue

BLOCKS = ("text", "image", "brand", "price", "list_price", "colour", "rating", "weight")
# Serving artifact: the chosen encoder's embeddings, already PCA-reduced (tools/export_embeddings.py)
IMAGE_FILE = Path(__file__).resolve().parent.parent / "data" / "image_embeddings.npz"
IMAGE_DIMS = 64  # PCA target: centres the features and keeps one query under ~1 ms (D11)
log = logging.getLogger("simsearch")
NEIGHBOUR_CREDIT = 0.5  # adjacent bands count as half a match (D8)


def multi_hot(labels: List[List[str]]) -> sparse.csr_matrix:
    """One column per distinct label; rows with no label stay all-zero (unknown never matches)."""
    vocab = {label: i for i, label in enumerate(sorted({label for row in labels for label in row}))}
    rows = [r for r, row in enumerate(labels) for _ in row]
    cols = [vocab[label] for row in labels for label in row]
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                             shape=(len(labels), max(len(vocab), 1)))


def soft_bands(bands: List[Optional[int]], n_bands: int) -> sparse.csr_matrix:
    """Ordinal bands with partial credit for neighbours: after normalisation same band = 1.0,
    adjacent = 0.67, two apart = 0.17. None -> all-zero row."""
    rows, cols, data = [], [], []
    for r, b in enumerate(bands):
        if b is None:
            continue
        for offset, credit in ((-1, NEIGHBOUR_CREDIT), (0, 1.0), (1, NEIGHBOUR_CREDIT)):
            if 0 <= b + offset < n_bands:
                rows.append(r)
                cols.append(b + offset)
                data.append(credit)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(bands), n_bands), dtype=np.float32)


def quantile_bands(values: List[Optional[float]], n: int) -> sparse.csr_matrix:
    """Equal-population bands: robust to the heavy skew in prices and weights (D8)."""
    known = np.array([v for v in values if v is not None], dtype=float)
    if known.size == 0:  # nothing known -> no signal from this block
        return soft_bands([None] * len(values), n)
    edges = np.quantile(known, np.linspace(0, 1, n + 1)[1:-1])
    return soft_bands([None if v is None else int(np.searchsorted(edges, v)) for v in values], n)


def reduce_embeddings(feats: np.ndarray, dims: int = IMAGE_DIMS) -> np.ndarray:
    """PCA to `dims`: removes the shared mean direction (raw CNN features are all-positive, so raw
    cosines bunch near 1) and shrinks the block so queries stay fast."""
    if feats.shape[1] <= dims or len(feats) <= dims:
        return feats
    return PCA(n_components=dims, random_state=0).fit_transform(feats)


def image_block(cat: Catalogue, path: Path = IMAGE_FILE, dims: int = IMAGE_DIMS) -> sparse.csr_matrix:
    """Image embeddings aligned to the catalogue. Fallback (D12): a product without an image, or a
    missing embeddings file, gets an all-zero row, so ranking falls back to the other blocks."""
    out = np.zeros((len(cat), dims), dtype=np.float32)
    if not path.exists():
        log.warning("image embeddings not found at %s; image block disabled (run tools/ to build it)", path)
        return sparse.csr_matrix(out)
    data = np.load(path)
    row = {pid: i for i, pid in enumerate(data["ids"])}
    idx = np.array([row.get(pid, -1) for pid in cat.ids])
    present = idx >= 0
    present[present] = data["has_image"][idx[present]]
    if present.any():
        reduced = reduce_embeddings(data["embeddings"][idx[present]].astype(np.float32), dims)
        out = np.zeros((len(cat), reduced.shape[1]), dtype=np.float32)
        out[present] = reduced
    return sparse.csr_matrix(out)


def build_blocks(cat: Catalogue, image_file: Path = IMAGE_FILE) -> Dict[str, sparse.csr_matrix]:
    raw = {
        "text": TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(cat.text),
        "image": image_block(cat, image_file),
        "list_price": quantile_bands(cat.list_prices, 10),
        "brand": multi_hot([[b] if b else [] for b in cat.brands]),
        "colour": multi_hot(cat.colours),
        "price": quantile_bands(cat.prices, 10),
        # ratings 1.0-5.0 in half-star steps -> bands 0..8
        "rating": soft_bands([None if r is None else int(round((r - 1) * 2)) for r in cat.ratings], 9),
        "weight": quantile_bands(cat.weights_g, 5),
    }
    return {name: normalize(raw[name]).astype(np.float32).tocsr() for name in BLOCKS}


def combine(blocks: Dict[str, sparse.csr_matrix], weights: Dict[str, float]) -> sparse.csr_matrix:
    """Scale each block by sqrt(weight) and concatenate: dot product = sum_b w_b * cos_b (D6)."""
    total = sum(weights.get(b, 0.0) for b in BLOCKS)
    return sparse.hstack([blocks[b] * np.float32(np.sqrt(weights.get(b, 0.0) / total)) for b in BLOCKS]).tocsr()
