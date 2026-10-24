"""Build docs/index.html: the served model's answers for every product in one page that opens with no server.

For all 30,000 products it stores the served top 10 (with each result's per-attribute contribution) and the
text-only baseline's top 10, packed as binary arrays inside the page. Before writing, it checks the stored lists
against find_similar_products on 300 random products and stops on any mismatch, so the page can't drift from the
engine. A test checks that the page's model version matches config/manifest.json. Photos load from the catalogue's
image links, so they need an internet connection.

    python tools/export_static_demo.py   -> docs/index.html (about 10 MB)
"""
import base64
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.data import load_catalogue  # noqa: E402
from simsearch.engine import SimilarityEngine, find_similar_products, load_model  # noqa: E402
from simsearch.features import BLOCKS, build_blocks  # noqa: E402
from simsearch.manifest import verify  # noqa: E402

K = 10
PREFIX = "https://images-na.ssl-images-amazon.com/images/I/"
OUT = ROOT / "docs" / "index.html"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main():
    cat = load_catalogue()
    blocks, model = build_blocks(cat), load_model()
    policy = model["policy"]
    served = SimilarityEngine(cat, weights=model["weights"], blocks=blocks, category_penalty=policy["category_penalty"],
                              name_penalty=policy.get("name_penalty", 0.0))
    text = SimilarityEngine(cat, weights={b: float(b == "text") for b in BLOCKS}, blocks=blocks)
    n = len(cat)
    top, base = np.zeros((n, K), np.uint16), np.zeros((n, K), np.uint16)
    score = np.zeros((n, K), np.int16)                       # score x 10,000
    parts = np.zeros((n, K, len(BLOCKS) + 1), np.uint8)      # each block's contribution x 250, then the penalty
    for q in range(n):
        res = served.similar_indices(q, K)
        top[q, :len(res)] = res
        base[q] = text.similar_indices(q, K)[:K]
        for j, row in enumerate(served.explain(q, res)):
            score[q, j] = round(sum(row.values()) * 10000)
            parts[q, j, :len(BLOCKS)] = [min(255, round(max(row[b], 0) * 250)) for b in BLOCKS]
            parts[q, j, -1] = min(255, round(-row["category"] * 250))

    sample = random.Random(1).sample(range(n), 300)
    wrong = [q for q in sample if [cat.ids[i] for i in top[q]] != find_similar_products(cat.ids[q], K)]
    if wrong:
        raise SystemExit(f"stored lists differ from the engine for {len(wrong)} of 300 products; not writing")

    brands = sorted({b for b in cat.brand_names if b})
    cats = sorted({c for c in cat.categories if c})
    bi, ci = {b: i + 1 for i, b in enumerate(brands)}, {c: i + 1 for i, c in enumerate(cats)}
    data = {
        "version": verify()["model_version"], "k": K, "blocks": BLOCKS, "weights": model["weights"],
        "penalty": policy["category_penalty"], "imgPrefix": PREFIX, "ids": cat.ids, "names": cat.names,
        "brandList": brands, "brand": [bi.get(b, 0) for b in cat.brand_names],
        "catList": cats, "cat": [ci.get(c, 0) for c in cat.categories],
        "price": [round(p) if p else 0 for p in cat.prices], "rating": [round(r * 10) if r else 0 for r in cat.ratings],
        "img": [u[len(PREFIX):] if u.startswith(PREFIX) else u for u in cat.image_urls],
        "top": b64(top), "score": b64(score), "parts": b64(parts), "base": b64(base),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    page = (ROOT / "web" / "static.html").read_text(encoding="utf-8").replace("__DATA__", payload)
    OUT.write_text(page, encoding="utf-8", newline="\n")
    print(f"{OUT} {OUT.stat().st_size / 1e6:.1f} MB, model {data['version']}, 300/300 lists match the engine")


if __name__ == "__main__":
    main()
