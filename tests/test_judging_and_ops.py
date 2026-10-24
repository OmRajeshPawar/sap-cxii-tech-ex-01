"""The model manifest, the static demo's version and the HNSW index."""
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from simsearch import manifest
from simsearch.ann import HnswIndex


def test_manifest_hash_ignores_line_endings_and_detects_tampering(monkeypatch):
    class Text:   # a stand-in file: .json content with Windows or Unix line endings
        suffix = ".json"

        def __init__(self, data):
            self.data = data

        def read_bytes(self):
            return self.data
    assert manifest.sha256(Text(b'{"a": 1}\r\n')) == manifest.sha256(Text(b'{"a": 1}\n'))
    good = manifest.current_hashes()
    assert manifest.verify()["verified"]
    monkeypatch.setattr(manifest, "current_hashes", lambda: {**good, "model_config": "0" * 64})
    tampered = manifest.verify()
    assert not tampered["verified"] and tampered["mismatched"] == ["model_config"]
    assert tampered["model_version"] != tampered["manifest_version"]


@pytest.mark.parametrize("seed", [0, 1])
def test_hnsw_candidates_cover_the_exact_neighbours(seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(20, 64))
    dense = np.repeat(centres, 50, axis=0) + 0.1 * rng.normal(size=(1000, 64))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    index = HnswIndex(sparse.csr_matrix(dense.astype(np.float32)), dims=32)
    recall = []
    for q in rng.choice(1000, 50, replace=False):
        exact = set(np.argsort(-(dense @ dense[q]))[:10])
        recall.append(len(exact & set(index.candidates(q, 50))) / 10)
    assert np.mean(recall) > 0.95


def test_static_demo_page_matches_the_served_model_version():
    """docs/index.html stores precomputed answers; it must be regenerated whenever the model version changes."""
    page = Path(__file__).resolve().parent.parent / "docs" / "index.html"
    if not page.exists():
        pytest.skip("static demo not generated")
    text = page.read_text(encoding="utf-8")
    stored = re.search(r'"version":"(\w+)"', text).group(1)
    assert stored == manifest.verify()["model_version"], "run python tools/export_static_demo.py"
