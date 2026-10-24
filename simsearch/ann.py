"""Bonus: approximate nearest-neighbour candidates with HNSW (D19).

HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
Hierarchical Navigable Small World graphs", IEEE TPAMI 2020 (arXiv:1603.09320).
HNSW needs dense vectors, so the sparse weighted matrix is first compressed with truncated SVD
(latent semantic analysis); HNSW then proposes candidates that are re-scored exactly.
"""
import faiss
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

SVD_DIMS = 256      # recall is limited by the SVD, not HNSW: more dims raised it, efSearch did not (D19)
HNSW_M = 32         # graph degree: higher = better recall, more memory
EF_CONSTRUCTION = 200
EF_SEARCH = 200     # search breadth; beyond the candidate pool size it made no measurable difference
CANDIDATE_POOL = 200  # candidates re-scored exactly; measured recall vs exact: results/evaluation.json


class HnswIndex:
    def __init__(self, matrix: sparse.csr_matrix, dims: int = SVD_DIMS, seed: int = 0):
        dims = min(dims, matrix.shape[1] - 1)   # tiny catalogues (tests) have fewer features than SVD_DIMS
        svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.vectors = np.ascontiguousarray(normalize(svd.fit_transform(matrix)).astype(np.float32))
        self.index = faiss.IndexHNSWFlat(self.vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = EF_CONSTRUCTION
        self.index.add(self.vectors)
        self.index.hnsw.efSearch = EF_SEARCH

    def candidates(self, q: int, n: int) -> np.ndarray:
        _, ids = self.index.search(self.vectors[q:q + 1], n)
        return ids[0][ids[0] >= 0]
