"""Feature blocks (simsearch/features.py): bands, multi-hot, photo alignment and the weighted combination."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import normalize

from conftest import product
from simsearch import features
from simsearch.data import build_catalogue
from simsearch.features import BLOCKS, build_blocks, combine, multi_hot, quantile_bands, reduce_embeddings


def test_quantile_bands_split_skewed_values_into_equal_groups_and_skip_unknowns():
    prices = [100, 120, 150, 200, 300, 500, 900, 2000, 9000, 30000, None]   # heavily skewed, one unknown
    m = quantile_bands(prices, 5).toarray()
    centre = m.argmax(axis=1)[:-1]
    assert (np.diff(centre) >= 0).all()                   # bands follow price order
    assert np.bincount(centre, minlength=5).max() <= 3    # ~equal population despite the skew
    assert not m[-1].any()                                # unknown price: empty row, never a match
    assert not quantile_bands([None, None], 4).toarray().any()


def test_multi_hot_leaves_unknown_rows_empty():
    m = multi_hot([["black", "white"], [], ["black"]]).toarray()
    assert m.sum(axis=1).tolist() == [2, 0, 1] and m.shape[1] == 2


def test_reduce_embeddings_centres_and_keeps_small_inputs_unchanged():
    feats = np.abs(np.random.default_rng(0).normal(size=(200, 120))) + 5   # all-positive, like CNN features
    reduced = reduce_embeddings(feats, 16)
    assert reduced.shape == (200, 16) and np.allclose(reduced.mean(axis=0), 0, atol=1e-6)
    small = feats[:, :8]
    assert reduce_embeddings(small, 16) is small


def test_photo_block_aligns_by_product_id_and_falls_back_for_missing_photos(monkeypatch):
    cat = build_catalogue([product("a", "red shoe"), product("b", "blue shoe"), product("c", "green hat")])
    saved = {"ids": np.array(["c", "a", "zz"]), "has_image": np.array([True, False, True]),
             "embeddings": np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float32)}
    monkeypatch.setattr(features.np, "load", lambda path: saved)
    block = features.image_block(cat, Path(__file__), dims=2).toarray()
    assert block[2].tolist() == [0, 1]                    # "c" gets its own row, matched by id not position
    assert not block[0].any() and not block[1].any()      # "a" has no usable photo, "b" is absent: zero rows


@pytest.mark.parametrize("weights", [{"text": 1.0}, {"text": 0.3, "brand": 0.7}, {b: 1.0 for b in BLOCKS}])
def test_combined_dot_product_equals_the_weighted_sum_of_block_cosines(weights):
    cat = build_catalogue([product("a", "cotton kurta", brand_="Biba", colour="Red", price="500"),
                           product("b", "cotton kurta long", brand_="Biba", colour="Blue", price="700"),
                           product("c", "leather wallet", brand_="Hide", price="2500")])
    blocks = build_blocks(cat, Path("no/such/embeddings.npz"))
    m = combine(blocks, weights).toarray()
    total = sum(weights.values())
    expected = sum(w / total * (blocks[b].toarray() @ blocks[b].toarray().T) for b, w in weights.items())
    assert np.allclose(m @ m.T, expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(normalize(blocks["text"]).toarray(), axis=1), 1)
