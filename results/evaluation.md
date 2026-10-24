Test set: 6561 queries that share no product (name, family or photo) with any tuning, validation or early-check query. Co-purchase labels at product-family level, top-10. Gap = served minus row, nDCG@10, 95% paired-bootstrap CI.

| System | nDCG@10 | Recall@10 | HitRate@10 | MRR@10 | Same category | Gap to served |
|---|---|---|---|---|---|---|
| Random, same leaf category | 0.027 | 0.049 | 0.092 | 0.029 | 1.000 | - |
| Text only (TF-IDF) | 0.313 | 0.408 | 0.562 | 0.348 | 0.777 | +0.066 [+0.061, +0.071] |
| Hand-set weights (otherwise as served) | 0.353 | 0.453 | 0.611 | 0.393 | 0.943 | +0.026 [+0.023, +0.030] |
| Learned weights, exact-name de-duplication only | 0.383 | 0.493 | 0.651 | 0.423 | 0.836 | -0.005 [-0.007, -0.002] |
| + one listing per product (name, family, photo) | 0.386 | 0.496 | 0.654 | 0.425 | 0.836 | -0.007 [-0.010, -0.005] |
| + category penalty | 0.379 | 0.484 | 0.643 | 0.418 | 0.963 | +0.000 [-0.000, +0.001] |
| **+ name words where a category is missing = served** | 0.379 | 0.484 | 0.644 | 0.419 | 0.963 | - |

| Block switched off (served model) | nDCG@10 lost (95% CI) | Learned weight |
|---|---|---|
| text | +0.088 [+0.082, +0.094] | 0.344 |
| image | +0.038 [+0.034, +0.043] | 0.401 |
| brand | +0.008 [+0.006, +0.010] | 0.102 |
| price | +0.007 [+0.005, +0.008] | 0.045 |
| colour | +0.002 [+0.001, +0.003] | 0.062 |
| rating | +0.001 [-0.001, +0.003] | 0.033 |
| list_price | +0.000 [-0.000, +0.000] | 0.001 |
| weight | -0.001 [-0.001, -0.000] | 0.012 |

| Search (in-process, one query) | Median / p95 | Recall@10 vs exact | Index build |
|---|---|---|---|
| Exact (default) | 1.12 / 1.36 ms | 100% | none |
| HNSW, SVD-256 + exact re-rank (`SEARCH_MODE=hnsw`) | 1.05 / 1.26 ms | 96.2% | 7 s |

Without the D13 filter, 28.5% of a plain top-10 is the query itself or another listing of a product already shown (same name, family or photo). Served results contain none. Another colour of the same design is kept as an alternative (D13); some readers would still count it as a duplicate.

Start-up (load data, build features, build engine): 3.4 s. A repeated default request is served from the in-process cache in 1.2 µs.
