"""Learn the block weights from co-purchase weak labels on the tuning queries only (D18).

Labels and grouping are at exact-name level here (engine.group_by_name), the grain the co-purchase labels are
recorded at, so it reproduces config/model.json. Any "policy" already in the file is kept.

    python tune.py   -> config/model.json (weights + full provenance)
"""
import json
import time

import numpy as np

from simsearch.data import load_catalogue
from simsearch.engine import MODEL_FILE, PRIOR_WEIGHTS, SimilarityEngine, group_by_name, load_model
from simsearch.evaluation import K, REFINE_CONCENTRATION, REFINE_TRIALS, candidate_weights, tune_weights
from simsearch.features import IMAGE_FILE
from simsearch.labels import build_labels, query_splits, train_test_split

N_TRIALS, SEED = 400, 0


def main():
    start = time.perf_counter()
    cat = load_catalogue()
    labels = build_labels(cat)
    train, test = train_test_split(labels, seed=SEED)
    sample, _, _ = query_splits(labels, seed=SEED)
    trials = candidate_weights(N_TRIALS, np.random.default_rng(SEED), PRIOR_WEIGHTS)

    weights, ndcg = tune_weights(group_by_name(SimilarityEngine(cat, weights=PRIOR_WEIGHTS)), sample, labels, trials)
    policy = load_model().get("policy")
    encoder_meta = IMAGE_FILE.with_suffix(".json")
    MODEL_FILE.parent.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps({
        "weights": weights,
        "image_encoder": json.loads(encoder_meta.read_text())["description"] if encoder_meta.exists() else None,
        "objective": f"mean nDCG@{K} on {len(sample)} tuning queries (co-purchase weak labels)",
        "train_ndcg": {"learned": round(float(ndcg.max()), 4), "prior": round(float(ndcg[0]), 4),
                       "equal": round(float(ndcg[1]), 4), "text_only": round(float(ndcg[2]), 4)},
        "search": {"method": "coarse-to-fine random search on the weight simplex: 3 anchors + Dirichlet samples, "
                             "then Dirichlet samples centred on the best (concentration "
                             f"{REFINE_CONCENTRATION:g})",
                   "coarse_trials": len(trials), "refine_trials": REFINE_TRIALS, "seed": SEED,
                   "coarse_best_ndcg": round(float(ndcg[:len(trials)].max()), 4)},
        "data": {"labelled_queries": len(labels), "train_queries": len(train), "test_queries": len(test)},
        **({"policy": policy} if policy else {}),
    }, indent=2))
    print(f"train nDCG@{K}: prior {ndcg[0]:.4f} | equal {ndcg[1]:.4f} | text-only {ndcg[2]:.4f} | "
          f"learned {ndcg.max():.4f}\nweights: {weights}\n{time.perf_counter() - start:.0f}s -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
