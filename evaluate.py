"""Offline evaluation on fresh test queries (D20): co-purchase proxy metrics, baselines, ablation, serving.

Test queries share no product group (same name, family or photo) with any tuning, validation or early-check query:
tuning, validation, or the early check sample. Labels are co-purchases at product-family level
(labels.build_family_labels). The hand-labelled check (is the result a useful alternative?) is tools/human_check.py.

    python evaluate.py   -> results/evaluation.json and results/evaluation.md
"""
import json
import sys
import random
import re
import time
from pathlib import Path

import numpy as np

from simsearch.data import load_catalogue
from simsearch.engine import PRIOR_WEIGHTS, SimilarityEngine, group_by_name, load_model
from simsearch.evaluation import K, evaluate, paired_bootstrap, same_category_baseline
from simsearch.features import BLOCKS, build_blocks
from simsearch.labels import build_family_labels, build_labels, fresh_queries, query_splits

SEED = 0
OUT = Path(__file__).resolve().parent / "results" / "evaluation.json"


def latency(engine, queries):
    lat = []
    for q in queries:
        t = time.perf_counter()
        engine.similar_indices(q, K)
        lat.append((time.perf_counter() - t) * 1000)
    return float(np.median(lat)), float(np.percentile(lat, 95))


def repeat_share(engine, queries) -> float:
    """Share of a plain top-10 (no D13 filter) that is the query again or another listing of a product
    already shown (same name, family or photo)."""
    groups, repeats = engine.groups, []
    for q in queries:
        scores = engine.scores(q)
        top = engine._ordered(scores, np.argpartition(scores, -(K + 1))[-(K + 1):])
        top = [i for i in top if i != q][:K]
        seen = {groups[q]}
        for i in top:
            repeats.append(groups[i] in seen)
            seen.add(groups[i])
    return float(np.mean(repeats))


def label_stats(cat, labels) -> dict:
    """How the co-purchase labels behave: same category? lexical overlap? (README limitations, D16)."""
    first = {}
    for i, key in enumerate(cat.family_keys):   # labels are family keys
        first.setdefault(key, i)
    words = [set(re.findall(r"[a-z0-9]+", n.lower())) for n in cat.names]
    jaccard = lambda a, b: len(words[a] & words[b]) / max(1, len(words[a] | words[b]))  # noqa: E731
    pairs = [(q, first[k]) for q, keys in labels.items() for k in keys]
    rng = random.Random(0)
    random_pairs = [(rng.randrange(len(cat)), rng.randrange(len(cat))) for _ in range(20000)]
    return {"labelled_queries": len(labels), "pairs": len(pairs),
            "mean_relevant_per_query": float(np.mean([len(v) for v in labels.values()])),
            "pairs_same_leaf_category": float(np.mean([cat.categories[a] == cat.categories[b] for a, b in pairs])),
            "pairs_name_jaccard_ge_0.5": float(np.mean([jaccard(a, b) >= 0.5 for a, b in pairs])),
            "random_pairs_name_jaccard_ge_0.5": float(np.mean([jaccard(a, b) >= 0.5 for a, b in random_pairs]))}


def main():
    t0 = time.perf_counter()
    cat = load_catalogue()
    blocks = build_blocks(cat)
    model = load_model()
    learned, penalty = model["weights"], model.get("policy", {}).get("category_penalty", 0.0)
    names = model.get("policy", {}).get("name_penalty", 0.0)
    startup_s = time.perf_counter() - t0
    t = time.perf_counter()
    served = SimilarityEngine(cat, weights=learned, blocks=blocks, category_penalty=penalty, name_penalty=names)
    startup_s += time.perf_counter() - t

    fam, name_labels = build_family_labels(cat), build_labels(cat)
    tune, val, check = query_splits(name_labels, seed=SEED)
    queries = fresh_queries(fam, served.groups, tune + val + check)
    text = {b: float(b == "text") for b in BLOCKS}

    # Baselines, then the served model built up one component at a time (an additive ablation)
    systems = {
        "text_only": SimilarityEngine(cat, weights=text, blocks=blocks),
        "hand_set_prior": SimilarityEngine(cat, weights=PRIOR_WEIGHTS, blocks=blocks, category_penalty=penalty,
                                           name_penalty=names),
        "weights_only": group_by_name(SimilarityEngine(cat, weights=learned, blocks=blocks)),
        "with_grouping": SimilarityEngine(cat, weights=learned, blocks=blocks),
        "with_category_penalty": SimilarityEngine(cat, weights=learned, blocks=blocks, category_penalty=penalty),
        "served": served,
    }
    rows = {"random_same_category": {**same_category_baseline(cat, queries, fam, level="family"), "same_category": 1.0,
                                     "coverage": None}}
    per_query = {}
    for name, engine in systems.items():
        rows[name], per_query[name] = evaluate(engine, queries, fam, per_query=True, level="family")
    significance = {name: paired_bootstrap(per_query["served"], per_query[name]) for name in systems if name != "served"}
    ablation = {}
    for b in BLOCKS:   # drop-one ablation on the served model: switch one block off, keep the other weights
        if learned.get(b, 0) > 0:
            _, nd = evaluate(SimilarityEngine(cat, weights={**learned, b: 0.0}, blocks=blocks,
                                              category_penalty=penalty, name_penalty=names), queries, fam,
                             per_query=True, level="family")
            ablation[b] = paired_bootstrap(per_query["served"], nd)

    sample = random.Random(1).sample(range(len(cat)), 500)
    for q in sample:   # warm the default-setting cache, then time cache hits
        served.find_similar_products(cat.ids[q], K)
    t = time.perf_counter()
    for q in sample:
        served.find_similar_products(cat.ids[q], K)
    cached_ms = (time.perf_counter() - t) / len(sample) * 1000
    t = time.perf_counter()
    ann = SimilarityEngine(cat, weights=learned, blocks=blocks, search="hnsw", category_penalty=penalty,
                           name_penalty=names)
    ann_build = time.perf_counter() - t
    exact_out = {q: served.similar_indices(q, K) for q in sample}
    recall = float(np.mean([len(set(ann.similar_indices(q, K)) & set(exact_out[q])) / K for q in sample]))
    serving = {"startup_s": startup_s, "cached_ms": cached_ms, "repeat_share_without_filter": repeat_share(served, sample),
               "exact_ms": latency(served, sample), "hnsw_ms": latency(ann, sample),
               "hnsw_recall_vs_exact": recall, "hnsw_build_s": ann_build,
               "note": "in-process engine time on one laptop CPU core, cache bypassed; HTTP numbers: tools/http_bench.py"}

    split = {"labelled_queries_name_level": len(name_labels),   # the grain the weights are tuned at
             "labelled_queries_family_level": len(fam),          # the grain results are evaluated at
             "used_in_development": {"tune": len(tune), "validation": len(val), "early_check": len(check)},
             "fresh_test": len(queries), "rule": "no product group (name, family or photo) shared with any used query",
             "products_without_category": sum(1 for c in cat.categories if not c) / len(cat)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"split": split, "labels": label_stats(cat, fam), "quality": rows,
                               "significance": significance, "ablation": ablation,
                               "serving": serving, "weights": learned, "category_penalty": penalty,
                               "name_penalty": names}, indent=2))
    OUT.with_suffix(".md").write_text(report(rows, significance, ablation, serving, split, learned), encoding="utf-8")
    print(OUT.with_suffix(".md").read_text(encoding="utf-8"))


def report(rows, significance, ablation, serving, split, weights) -> str:
    """The README's result tables, generated so the numbers can't drift from the JSON."""
    ci = lambda s: f"{s['delta']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"  # noqa: E731
    names = {"random_same_category": "Random, same leaf category", "text_only": "Text only (TF-IDF)",
             "hand_set_prior": "Hand-set weights (otherwise as served)",
             "weights_only": "Learned weights, exact-name de-duplication only",
             "with_grouping": "+ one listing per product (name, family, photo)",
             "with_category_penalty": "+ category penalty",
             "served": "**+ name words where a category is missing = served**"}
    out = [f"Test set: {split['fresh_test']} queries that share no product (name, family or photo) with any tuning, "
           f"validation or early-check query. Co-purchase labels at product-family level, top-{K}. Gap = served minus row, "
           f"nDCG@{K}, 95% paired-bootstrap CI.", "",
           "| System | nDCG@10 | Recall@10 | HitRate@10 | MRR@10 | Same category | Gap to served |",
           "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        sc = "-" if r["same_category"] is None else f"{r['same_category']:.3f}"
        gap = ci(significance[name]) if name in significance else "-"
        out.append(f"| {names[name]} | {r['ndcg']:.3f} | {r['recall']:.3f} | {r['hit_rate']:.3f} | {r['mrr']:.3f} | "
                   f"{sc} | {gap} |")
    out += ["", "| Block switched off (served model) | nDCG@10 lost (95% CI) | Learned weight |", "|---|---|---|"]
    for b, s in sorted(ablation.items(), key=lambda kv: -kv[1]["delta"]):
        out.append(f"| {b} | {ci(s)} | {weights[b]:.3f} |")
    exact, hnsw = serving["exact_ms"], serving["hnsw_ms"]
    out += ["", "| Search (in-process, one query) | Median / p95 | Recall@10 vs exact | Index build |", "|---|---|---|---|",
            f"| Exact (default) | {exact[0]:.2f} / {exact[1]:.2f} ms | 100% | none |",
            f"| HNSW, SVD-256 + exact re-rank (`SEARCH_MODE=hnsw`) | {hnsw[0]:.2f} / {hnsw[1]:.2f} ms | "
            f"{serving['hnsw_recall_vs_exact']:.1%} | {serving['hnsw_build_s']:.0f} s |",
            "", f"Without the D13 filter, {serving['repeat_share_without_filter']:.1%} of a plain top-{K} is the query "
            "again or another listing of a product already shown (same name, family or photo); served results contain "
            "none. Another colour of the same design is kept as an alternative (D13), which some readers would still call a "
            "duplicate.",
            "", f"Start-up (load data, build features, build engine): {serving['startup_s']:.1f} s. A repeated "
            f"default request is served from the in-process cache in {serving['cached_ms'] * 1000:.1f} µs."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    main()
