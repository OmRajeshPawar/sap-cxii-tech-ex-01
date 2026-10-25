"""Demo UI routes, kept separate from the brief's endpoint in app.py.

GET /                         -> web/index.html (single-file UI)
GET /api/products?q=&limit=   -> name search (every word must appear), or random picks when q is empty
GET /api/random               -> one random product with a photo ("Surprise me")
GET /api/similar/{product_id} -> query + results with per-block explanations and the signals used
GET /api/model                -> weights, policy and the saved evaluation results
GET /api/cases                -> the guided walkthrough cases (results/demo_cases.json)
"""
import json
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from simsearch.data import category_text
from simsearch.engine import MODEL_FILE, get_engine, parse_weights
from simsearch.features import BLOCKS

ROOT = Path(__file__).resolve().parent
router = APIRouter()


def card(i: int) -> dict:
    cat = get_engine().cat
    return {"id": cat.ids[i], "name": cat.names[i], "brand": cat.brand_names[i], "price": cat.prices[i],
            "list_price": cat.list_prices[i], "rating": cat.ratings[i], "image": cat.image_urls[i],
            "category": category_text({cat.categories[i]: ""}) if cat.categories[i] else None}


@router.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


@lru_cache(maxsize=1)
def search_index():
    """Lower-cased names and the products that have a photo, computed once per process."""
    cat = get_engine().cat
    return [n.lower() for n in cat.names], [i for i, u in enumerate(cat.image_urls) if u]


@router.get("/api/products")
def products(q: str = "", limit: int = Query(12, ge=1, le=50)):
    names, with_photo = search_index()
    if not q.strip():
        picks = random.sample(with_photo, limit)
        return {"query": "", "total": len(picks), "items": [card(i) for i in picks]}
    terms = q.lower().split()
    matches = [i for i, n in enumerate(names) if all(t in n for t in terms)]
    return {"query": q, "total": len(matches), "items": [card(i) for i in matches[:limit]]}


@router.get("/api/random")
def random_product():
    return card(random.choice(search_index()[1]))


@router.get("/api/similar/{product_id}")
def similar(product_id: str, num_similar: int = Query(10, ge=1), weights: Optional[str] = None,
            category_penalty: Optional[float] = Query(None, ge=0, le=1)):
    engine = get_engine()
    if product_id not in engine.index:   # checked explicitly, like the main endpoint
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    start = time.perf_counter()
    try:
        custom = parse_weights(weights)   # same validation as the main endpoint
        ids = engine.find_similar_products(product_id, num_similar, custom, category_penalty)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    took_ms = (time.perf_counter() - start) * 1000
    q = engine.index[product_id]
    idx = [engine.index[i] for i in ids]
    parts = engine.explain(q, idx, custom, category_penalty)
    return {"query": card(q), "blocks": list(BLOCKS), "signal": engine.signal(q, custom), "took_ms": took_ms,
            "category_penalty": engine.category_penalty if category_penalty is None else category_penalty,
            "results": [{**card(i), "score": sum(p.values()), "contributions": p,
                         "same_type": bool(engine.category_code[i] == engine.category_code[q])
                         if engine.category_code[q] >= 0 and engine.category_code[i] >= 0 else None}
                        for i, p in zip(idx, parts)]}


@router.get("/api/model")
def model():
    info = json.loads(MODEL_FILE.read_text()) if MODEL_FILE.exists() else {}
    evaluation, human = read_result("evaluation.json"), read_result("human_check/human_eval.json")
    return {"weights": get_engine().weights, "policy": info.get("policy", {}), "image_encoder": info.get("image_encoder"),
            "evaluation": {k: evaluation.get(k) for k in ("split", "quality", "significance", "ablation")},
            "human": human}


@router.get("/api/cases")
def cases():
    return read_result("demo_cases.json").get("cases", [])


def read_result(name: str) -> dict:
    """A results/ file written by an evaluation script; empty if not generated yet."""
    path = ROOT / "results" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
