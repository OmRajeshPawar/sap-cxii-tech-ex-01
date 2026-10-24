"""Scoring, ranking and request validation (simsearch/engine.py, features.py, ann.py)."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import normalize

from conftest import engine_for, product
from simsearch.data import build_catalogue
from simsearch.engine import PRIOR_WEIGHTS, SimilarityEngine, product_groups
from simsearch.features import combine, image_block, soft_bands


def three_products():
    return engine_for([product("q", "cotton kurta", brand_="Biba", price="500"),
                       product("same_brand", "silk saree", brand_="Biba", price="5000"),
                       product("same_text", "cotton kurta straight", brand_="Aurelia", price="5000")])


def test_per_request_weights_change_ranking():
    engine = three_products()
    assert engine.find_similar_products("q", 1, {"brand": 1.0}) == ["same_brand"]
    assert engine.find_similar_products("q", 1, {"text": 1.0}) == ["same_text"]


@pytest.mark.parametrize("bad", [{"colorr": 1.0}, {"text": -1.0}, {"text": 0.0}, {"text": float("nan")},
                                 {"text": 1e308, "brand": 1e308}])   # the last one overflows to inf when summed
def test_invalid_request_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        three_products().find_similar_products("q", 1, bad)


@pytest.mark.parametrize("bad", [{"text": 0.0}, {"text": 1.0, "image": -0.5}])
def test_invalid_engine_weights_are_rejected(bad):
    engine = three_products()
    with pytest.raises(ValueError):
        SimilarityEngine(engine.cat, weights=bad, blocks=engine.blocks)


@pytest.mark.parametrize("k, penalty", [(0, None), (-1, None), (1, 1.5), (1, -0.1)])
def test_invalid_k_or_penalty_is_rejected(k, penalty):
    with pytest.raises(ValueError):
        three_products().find_similar_products("q", k, category_penalty=penalty)


def test_fast_scoring_matches_the_weighted_cosine_definition():
    engine = engine_for([product("a", "red cotton kurta", brand_="Biba", colour="Red", price="500"),
                         product("b", "blue cotton kurta", brand_="Biba", colour="Blue", price="700"),
                         product("c", "leather wallet", brand_="Hidesign", price="2500")])
    reference = combine(engine.blocks, engine.weights)   # one matrix: dot product = sum_b w_b * cos_b
    for q in range(3):
        assert np.allclose(engine.scores(q), reference @ reference[q].toarray().ravel(), atol=1e-6)


def test_soft_bands_give_partial_credit_and_unknown_never_matches():
    m = normalize(soft_bands([3, 3, 4, 5, None], 10)).toarray()
    assert np.isclose(m[0] @ m[1], 1.0) and 0.6 < m[0] @ m[2] < 0.7 and 0.1 < m[0] @ m[3] < 0.2
    assert not m[4].any()


def test_missing_image_embeddings_fall_back_to_zero_block():
    cat = build_catalogue([product("a", "red shoe"), product("b", "blue shoe")])
    block = image_block(cat, Path("no/such/embeddings.npz"))
    assert block.shape[0] == 2 and block.nnz == 0   # no crash, image simply contributes nothing


def test_request_with_no_usable_signal_is_rejected():
    engine = engine_for([product("q", "red kurta", brand_=""), product("b", "red kurta long", brand_="Biba")])
    with pytest.raises(ValueError, match="no data"):
        engine.find_similar_products("q", 1, {"brand": 1.0})   # q has no brand, so every score would be 0
    assert engine.signal(0, {"brand": 0.5, "text": 0.5}) == {"available": ["text"], "missing": ["brand"]}


def test_ranking_skips_query_duplicates_and_variants_deterministically():
    engine = engine_for([
        product("q", "linen summer shirt", brand_="Zara", colour="White"),
        product("dup", "linen summer shirt", brand_="Zara", asin="q"),          # same listing twice
        product("x", "linen summer shirt slim", brand_="Zara", colour="White", rating="4.5"),
        product("x2", "Linen Summer Shirt Slim", brand_="Zara", colour="White"),  # variant of x
        product("z", "linen summer shirt regular", brand_="Zara", rating="4.5"),
        product("far", "silk saree blouse", brand_="Biba", price="5000"),
    ])
    result = engine.find_similar_products("q", 3)
    assert result[0] in {"x", "x2"}                           # closest match first (one of the variants)
    assert "dup" not in result and not {"x", "x2"} <= set(result)
    assert result == engine.find_similar_products("q", 3)       # same input, same order


def test_identical_photos_and_size_variants_are_shown_once():
    cat = build_catalogue([product("q", "linen shirt"), product("tee_s", "cotton tee (S)"), product("tee_xl", "cotton tee (XL)"),
                           product("p1", "polo one"), product("p2", "polo two")])
    image = np.zeros((5, 4), dtype=np.float32)
    image[3] = image[4] = [1, 0, 0, 0]   # p1 and p2 are the same photo listed twice
    groups = product_groups(cat, image)
    assert groups[1] == groups[2] and groups[3] == groups[4] and len(set(groups)) == 3
    engine = engine_for([product("q", "linen shirt"), product("tee_s", "cotton tee (S)"), product("tee_xl", "cotton tee (XL)")])
    assert len(engine.find_similar_products("q", 5)) == 1   # two sizes of one tee are one product


def test_type_penalty_demotes_other_categories_and_is_explained():
    products = [product("q", "striped cotton t shirt", brand_="Ruf", category="MensTShirts"),
                product("jeans", "striped cotton t shirt jeans", brand_="Ruf", category="MensJeans"),
                product("tee", "striped t shirt", brand_="Other", category="MensTShirts"),
                product("unknown", "striped cotton shirt", brand_="Ruf", category=None)]
    cat = build_catalogue(products)
    plain = SimilarityEngine(cat, weights=dict(PRIOR_WEIGHTS))
    strict = SimilarityEngine(cat, weights=dict(PRIOR_WEIGHTS), blocks=plain.blocks, category_penalty=0.5)
    assert plain.find_similar_products("q", 1) == ["jeans"]
    assert strict.find_similar_products("q", 1) != ["jeans"]
    assert strict.type_penalty(0)[3] == 0                     # an unknown category is never penalised
    q, res = 0, [strict.index[i] for i in strict.find_similar_products("q", 3)]
    parts = strict.explain(q, res)
    assert parts[res.index(1)]["category"] == -0.5            # the penalty is visible in the explanation
    assert plain.find_similar_products("q", 1, category_penalty=0.5) == strict.find_similar_products("q", 1)


def test_name_words_stand_in_for_a_missing_category():
    products = [product("q", "Men's Cotton T-Shirt Navy", brand_="Max", category=None),
                product("womens", "Women's Cotton T-Shirt Navy", brand_="Max", category=None),
                product("jeans", "Men's Cotton Jeans Navy", brand_="Max", category=None),
                product("mens", "Men's Cotton T-Shirt Grey", brand_="Puma", category=None),
                product("known", "Women's Cotton T-Shirt Blue", brand_="Max", category="WomensTops")]
    cat = build_catalogue(products)
    plain = SimilarityEngine(cat, weights=dict(PRIOR_WEIGHTS), category_penalty=0.15)
    rule = SimilarityEngine(cat, weights=dict(PRIOR_WEIGHTS), blocks=plain.blocks, category_penalty=0.15,
                            name_penalty=0.1)
    penalty = dict(zip(cat.ids, rule.type_penalty(0)))
    assert penalty["womens"] == penalty["jeans"] == penalty["known"] == 0.1   # other wearer, other type, by name
    assert penalty["mens"] == 0                                               # same wearer and type: no penalty
    before, after = plain.find_similar_products("q", 4), rule.find_similar_products("q", 4)
    assert before.index("mens") == 3 and after.index("mens") == 1   # the same wearer and type moves up past women's items
    assert after[0] == "jeans"                                      # a demotion, not a filter: a very close match can stay
    assert not rule.type_penalty(0, 0.0).any()                      # category_penalty=0 turns both rules off
    assert rule.find_similar_products("q", 4, category_penalty=0) == plain.find_similar_products("q", 4, category_penalty=0)
    both = SimilarityEngine(build_catalogue([product("a", "Men's Shirt", category="MensShirts"),
                                             product("b", "Women's Shirt", category="MensShirts")]),
                            weights=dict(PRIOR_WEIGHTS), category_penalty=0.15, name_penalty=0.1)
    assert both.type_penalty(0)[1] == 0      # both categories known: the category decides, names are ignored



def test_hnsw_mode_returns_distinct_results_and_survives_huge_k():
    products = [product("a", "red cotton kurta", brand_="Biba"), product("b", "red cotton kurta long", brand_="Biba"),
                product("c", "blue denim jeans"), product("d", "leather wallet", brand_="Hidesign")]
    exact = engine_for(products)
    ann = SimilarityEngine(exact.cat, weights=dict(PRIOR_WEIGHTS), search="hnsw")
    assert ann.find_similar_products("a", 1) == exact.find_similar_products("a", 1) == ["b"]
    assert sorted(ann.find_similar_products("a", 10**9)) == ["b", "c", "d"]   # capped at the catalogue
