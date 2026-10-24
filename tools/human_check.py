"""Score my blind hand labels: of the results I labelled, how many did I call real substitutes, per system?

    python tools/human_check.py   -> results/human_check/human_eval.json and human_eval.md

I labelled pairs (a query product and one result), shown in random order and pooled across the three systems, so I never
saw which system produced a result, its rank or its category. Labels: substitute (same kind of product for the same
wearer, so it could be bought instead), partial (same broad kind but a different wearer, sub-type or occasion, or a price
more than about 3x apart), duplicate (the query itself in another size, colour or pack) and not (another kind of product,
an accessory or a complement). The pairs come from a fixed random sample of held-out test queries (seed 11), top 5 of each
system. Because I labelled pairs and not whole lists, each list has only one or two labelled results, so the numbers are
shares of labelled results.

Keys of human_labels.json are "<query uniq_id>|<result uniq_id>".
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.data import load_catalogue  # noqa: E402
from simsearch.engine import SimilarityEngine, group_by_name, load_model  # noqa: E402
from simsearch.features import BLOCKS, build_blocks  # noqa: E402
from simsearch.labels import build_family_labels, build_labels, fresh_queries, query_splits  # noqa: E402

N_QUERIES, TOP, SEED, SAMPLE_SEED, TARGET_PAIRS = 150, 5, 0, 11, 1000
LABELS = ("substitute", "partial", "duplicate", "not")
OUT = ROOT / "results" / "human_check"
NAMES = {"text_only": "Text only (TF-IDF)", "weights_only": "Learned weights only", "served": "Served"}


def setup():
    """The catalogue, the three systems and the held-out test queries I sampled from."""
    cat = load_catalogue()
    blocks, model = build_blocks(cat), load_model()
    learned, policy = model["weights"], model["policy"]
    systems = {"text_only": SimilarityEngine(cat, weights={b: float(b == "text") for b in BLOCKS}, blocks=blocks),
               "weights_only": group_by_name(SimilarityEngine(cat, weights=learned, blocks=blocks)),
               "served": SimilarityEngine(cat, weights=learned, blocks=blocks, category_penalty=policy["category_penalty"],
                                          name_penalty=policy.get("name_penalty", 0.0))}
    tune, val, check = query_splits(build_labels(cat), seed=SEED)
    fresh = fresh_queries(build_family_labels(cat), systems["served"].groups, tune + val + check)
    return cat, systems, random.Random(SEED).sample(fresh, N_QUERIES)


def sample():
    """Test queries in a fixed random order until their pooled top-5 pairs reach TARGET_PAIRS."""
    cat, systems, test = setup()
    order = random.Random(SAMPLE_SEED).sample(test, len(test))
    tops = {n: {q: s.similar_indices(q, TOP) for q in order} for n, s in systems.items()}
    queries, pairs = [], set()
    for q in order:
        queries.append(q)
        pairs |= {(q, c) for n in NAMES for c in tops[n][q]}
        if len(pairs) >= TARGET_PAIRS:
            break
    return cat, queries, {n: {q: tops[n][q] for q in queries} for n in NAMES}


def main():
    cat, queries, top = sample()
    raw = json.loads((OUT / "human_labels.json").read_text(encoding="utf-8"))
    human = {tuple(k.split("|")): v for k, v in raw.items() if v in LABELS}
    lab = lambda q, c: human.get((cat.ids[q], cat.ids[c]))  # noqa: E731
    count = lambda n, ok: np.array([sum(lab(q, c) in ok for c in top[n][q]) for q in queries])  # noqa: E731
    den = {n: count(n, LABELS) for n in NAMES}
    sub, use = {n: count(n, ("substitute",)) for n in NAMES}, {n: count(n, ("substitute", "partial")) for n in NAMES}
    share = lambda num: {n: float(num[n].sum() / den[n].sum()) for n in NAMES}  # noqa: E731
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(queries), len(queries)) for _ in range(2000)]

    def diff(num, a, b):
        d = [num[a][i].sum() / den[a][i].sum() - num[b][i].sum() / den[b][i].sum() for i in boot]
        return {"delta": float(num[a].sum() / den[a].sum() - num[b].sum() / den[b].sum()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    result = {"labelled_pairs": len(human), "queries": int(sum(any(den[n][i] for n in NAMES) for i in range(len(queries)))),
              "labelled_results_per_system": {n: int(den[n].sum()) for n in NAMES},
              "substitute_share": share(sub), "substitute_or_partial_share": share(use),
              "served_minus_text_only": diff(sub, "served", "text_only"),
              "served_minus_weights_only": diff(sub, "served", "weights_only")}
    (OUT / "human_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    ci = lambda s: f"{100 * s['delta']:+.1f} [{100 * s['ci95'][0]:+.1f}, {100 * s['ci95'][1]:+.1f}]"  # noqa: E731
    lines = [f"I labelled {result['labelled_pairs']} result pairs from {result['queries']} held-out test queries, blind to the system, "
             "the rank and the category. Labels: substitute, partial, duplicate, not (definitions in `tools/human_check.py`).", "",
             "| System | Labelled results | Called a substitute | Substitute or partial |", "|---|---|---|---|"]
    lines += [f"| {NAMES[n]} | {result['labelled_results_per_system'][n]} | {100 * result['substitute_share'][n]:.1f}% | "
              f"{100 * result['substitute_or_partial_share'][n]:.1f}% |" for n in NAMES]
    lines += ["", f"Served minus text-only: {ci(result['served_minus_text_only'])} points. Served minus learned weights only: "
                  f"{ci(result['served_minus_weights_only'])} points. 95% bootstrap intervals over queries."]
    (OUT / "human_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
