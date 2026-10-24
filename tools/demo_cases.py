"""Pick the demo's guided-tour cases from saved data, deterministically, and write their explanations.

The cases come from the catalogue and the served engine.
  1. Why the type penalty exists: a striped T-shirt whose first "similar" product is jeans without it.
  2. A strong case: a held-out test query whose top 5 all share its category and which also hits a co-purchase
     label.
  3. Missing data: a listing with no photo, to show the fallback and the signals line.

    python tools/demo_cases.py   -> results/demo_cases.json
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.engine import SimilarityEngine, group_by_name, load_model  # noqa: E402
from simsearch.data import load_catalogue  # noqa: E402
from simsearch.features import BLOCKS, build_blocks  # noqa: E402
from simsearch.labels import build_family_labels, build_labels, fresh_queries, query_splits  # noqa: E402

N_QUERIES, SEED = 150, 0
TSHIRT = "2852dde03bc410bd2c8cf09db9ff674c"
NAMES = {"text": "text", "image": "photo", "brand": "brand", "price": "price", "list_price": "list price",
         "colour": "colour", "rating": "rating", "weight": "weight"}


def main():
    cat = load_catalogue()
    blocks, model = build_blocks(cat), load_model()
    learned, penalty = model["weights"], model["policy"]["category_penalty"]
    served = SimilarityEngine(cat, weights=learned, blocks=blocks, category_penalty=penalty,
                              name_penalty=model["policy"].get("name_penalty", 0.0))
    plain = group_by_name(SimilarityEngine(cat, weights=learned, blocks=blocks))   # no type penalty, names only
    short = lambda i: cat.names[i][:60]  # noqa: E731
    cases = []

    q = served.index[TSHIRT]
    before, after = plain.similar_indices(q, 5), served.similar_indices(q, 5)
    repeats = len(before) - len({served.groups[i] for i in before})
    cases.append({"id": TSHIRT, "title": "Why the type penalty exists",
                  "teaser": "A T-shirt that would recommend jeans",
                  "note": f"Without the type penalty, the first result for this striped T-shirt is \"{short(before[0])}\": "
                          f"shoppers buy them together, and the weights are learned from co-purchases. Grouping by "
                          f"exact name alone would also show {repeats} extra size{'s' if repeats != 1 else ''} of a "
                          f"product already listed. The served model subtracts {penalty:.2f} when a result's known "
                          f"category differs and shows one listing per product, so #1 is \"{short(after[0])}\". "
                          "Press \"No type penalty\" and the jeans come back to #1."})

    tune, val, check = query_splits(build_labels(cat), seed=SEED)
    fam = build_family_labels(cat)
    fresh = fresh_queries(fam, served.groups, tune + val + check)
    for q in random.Random(SEED).sample(fresh, N_QUERIES):
        top = served.similar_indices(q, 5)
        hits = [i for i in top if cat.family_keys[i] in fam[q]]
        if cat.image_urls[q] and cat.categories[q] and all(cat.categories[c] == cat.categories[q] for c in top) and hits:
            cases.append({"id": cat.ids[q], "title": "A strong match",
                          "teaser": "Five alternatives, one bought with it",
                          "note": f"A held-out test query (not used for tuning or validation). All five results are in the same "
                                  f"category as the query, and \"{short(hits[0])}\" is one shoppers actually "
                                  f"bought with it. The bars show the photo and the text doing most of the work."})
            break

    no_photo = [i for i in range(len(cat)) if not served.has_data[BLOCKS.index("image"), i] and cat.brands[i]
                and cat.categories[i] and cat.prices[i]]
    i = sorted(no_photo, key=lambda j: cat.ids[j])[0]
    has = [NAMES[b] for b in served.signal(i)["available"]]
    cases.append({"id": cat.ids[i], "title": "Missing data: no photo",
                  "teaser": "What happens when the strongest signal is absent",
                  "note": f"This listing has no photo, and the photo carries the largest weight ({learned['image']:.2f}). "
                          "Its image block is all zeros, so the photo adds nothing to any score and the ranking falls "
                          f"back to what it does have: {', '.join(has)}. Nothing is re-weighted, so a listing is never "
                          "rewarded for having less data. Move every weight except Image to 0: the API answers 422 "
                          "instead of returning products ranked on nothing."})

    (ROOT / "results" / "demo_cases.json").write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    for c in cases:
        print(f"{c['title']}: {c['note']}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    main()
