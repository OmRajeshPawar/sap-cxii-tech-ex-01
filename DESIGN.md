# Design decisions and trade-offs

The code cites these IDs in comments. Each entry gives the choice, the reason, the alternatives and the evidence.
Numbers come from `results/` and `analysis/`; a "development measurement" is a one-off comparison that informed a choice.

Weights are learned on 1,000 tuning queries. Every other choice (photo encoder, penalties, experiments) is made on
2,000 validation queries or on separate products, against a decision rule written in advance. Reported quality comes
from 6,561 test queries that share no product with any of them (D25).

## A. Data

The findings behind these decisions are in [analysis/data_report.md](analysis/data_report.md), produced by
`analysis/explore_data.py`.

| ID | Decision | Why, and what I rejected |
|---|---|---|
| D1 | I read straight from `data/archive.zip` | There is no extraction step to forget, and the container ships one file. |
| D2 | `weight = 999999999` means unknown, and real weights are converted to grams | 79% of rows hold that placeholder. Read as a number, it would make most products match on weight. Filling it with the median would invent data |
| D3 | Unknown never counts as a match | A missing brand, colour, price, weight or photo is an all-zero vector, so two unbranded products score 0 on brand, not 1. "Generic" and "Unknown" count as missing. An explicit "Unknown" value would make thousands of unbranded products look alike |
| D4 | Category text comes from the category keys, split on CamelCase | `WomensKurtasKurtis` becomes "Womens Kurtas Kurtis", which says what the product is. The values are sales ranks, which measure popularity |
| D5 | Two price blocks: `price` is what the shopper pays (`sales_price`), `list_price` is the original price derived from `discount_percentage` | The data has only `sales_price`, so the brief's separate *price* is recovered as the list price. The learned weight puts nearly everything on the price paid (0.04 vs 0.00), because the two move together |

The `colour` field holds the colours a product is sold in (median 3 per product), not the colour of the listing. I
compare those sets as a weak signal, and the weight search agreed it is weak: colour ends up at 0.06.

## B. Similarity model

| ID | Decision | Why, and what I rejected |
|---|---|---|
| D6 | A weighted per-attribute cosine, computed as one matrix | I L2-normalise each block and scale it by the square root of its weight, so one dot product equals the weighted sum of block cosines and every score splits exactly into per-attribute parts. A single cosine over all features would let the wide text block dominate. Gower distance needs pairwise loops. A neural ranker needs more labels than I have (D16) |
| D7 | TF-IDF on name and category | It is fast, deterministic and explainable. I tested sentence embeddings on validation (development measurement): blending FashionSigLIP's text embeddings into this block gained +0.004 nDCG@10, CI [−0.000, +0.008], under my 1% bar, and the semantic text alone scored 0.047 worse, because these labels reward exact wording. So the service needs no text model. `meta_keywords` repeats the name plus a few seller words, so I left it out |
| D8 | Prices, weights and ratings become equal-population bands, with partial credit for neighbours | Cosine on raw numbers measures direction, not closeness, and prices are heavily skewed. Neighbour credit keeps ₹499 and ₹501 close: the same band scores 1.0 and an adjacent band 0.67 |
| D9 | A hand-set prior is the fallback and the baseline | Learned weights beat it by +0.026 nDCG@10, CI [+0.023, +0.030], on the test set (`results/evaluation.md`) |

## C. Photos

D10. I chose Marqo-FashionSigLIP after comparing four encoders on the same photos, the same PCA and the same
validation queries, re-learning the weights for each (development measurement). The rule, fixed before running: the
highest full-model validation nDCG@10 wins, and a near-tie (within 1%) goes to the cheaper encoder.

| Encoder | Full model nDCG@10 | Gain over no photo (95% CI) | CPU time to embed 30k |
|---|---|---|---|
| No photo | 0.330 | - | - |
| ResNet-50 (the brief's example) | 0.355 | +0.025 [+0.017, +0.034] | 16 min |
| DINOv2 ViT-S/14 | 0.361 | +0.031 [+0.023, +0.039] | 16 min |
| SigLIP 2 ViT-B/16 | 0.377 | +0.047 [+0.038, +0.056] | 49 min |
| Marqo-FashionSigLIP | 0.383 | +0.053 [+0.046, +0.061] | 45 min |

FashionSigLIP beat SigLIP 2 by +0.006, CI [+0.001, +0.012], which is 1.7% relative. SigLIP 2 is better on photos
alone (0.187 against 0.179), but FashionSigLIP adds more of what the text block doesn't already capture, which is why
I compared encoders inside the full model. Both SigLIP models are listed as Apache-2.0 on their model cards.

| ID | Decision | Why, and what I rejected |
|---|---|---|
| D11 | Embed offline, reduce to 64 dimensions with PCA, ship a 4 MB file | The API never loads PyTorch. PCA also centres the features, since raw CNN cosines bunch near 1. Embedding at serving time would mean a large image and a slow start. t-SNE is a visualisation method: it can't place new products and its distances aren't meaningful for retrieval |
| D12 | A missing photo or attribute is a zero block, and scores are not re-weighted | The ranking falls back to the attributes that exist (139 products have no usable photo). Re-weighting each pair by the attributes both products have would over-reward sparse products |

## D. Ranking

| ID | Decision | Why, and what I rejected |
|---|---|---|
| D13 | One listing per real product | Listings that share a normalised name, a family key (the name without size words, age ranges, pack counts or bracketed codes) or a byte-identical photo are linked, and I show one listing per group. Without this, 28.5% of a plain top 10 repeats a product already shown (`results/evaluation.md`). Colour words stay in the key, so another colour remains a separate alternative. I measured and rejected two stronger rules: splitting identical titles by photo (size variants have different photos 69% of the time) and a near-duplicate photo threshold (photo cosine doesn't separate duplicates from substitutes) |
| D14 | Ties go to the higher rating, then the smaller id | The output is deterministic, which also makes it testable |
| D15 | Exact search, made fast | The dense 64-dimensional photo block is a plain dense product, and the sparse blocks read only the columns the query uses. `np.partition` picks the top candidates, and ids compare through a precomputed integer rank, because sorting id strings dominated query time when I profiled it. Precomputing all pairs would take about 3.6 GB |

## E. Learning and evaluation

| ID | Decision | Why, and what I rejected |
|---|---|---|
| D16 | Weak labels from `other_items_customers_buy` | There is no ground truth for "similar". Co-purchased names matched back to the catalogue give 12,193 labelled queries at name level (11,410 at product-family level). I never use them as a feature. They have two biases: they reward complements, and they lean on wording (31% of pairs share at least half their name words; `results/evaluation.json`, `labels`). That is why I added a hand-labelled check (D26) |
| D17 | Three fixed query sets, paired bootstrap CIs and a drop-one ablation | Tune 1,000, validation 2,000, and a fresh test set (D25). Every comparison is a paired bootstrap over queries with 2,000 resamples. Metrics: nDCG, Recall, HitRate and MRR at 10. Baselines: random products from the same category, text-only, hand-set weights. |
| D18 | Weights by coarse-to-fine random search on the simplex | 400 Dirichlet samples plus 3 anchors, then 200 samples around the best, scored by nDCG@10 on the tuning queries, in a few minutes on a CPU. The result: photo 0.40, text 0.34, brand 0.10, colour 0.06, price 0.04, rating 0.03, weight 0.01, list price 0.00. Grid search is too expensive in 8 dimensions. LambdaMART can't be supported by about 2 labels per query, and it would lose the explainable linear form |
| D24 | "Similar" means something to buy instead, enforced with a product-type penalty of λ = 0.15 | Co-purchase weights reward complements: a striped T-shirt's first result was jeans. When both leaf categories are known and differ, I subtract λ; an unknown category is never penalised. I chose λ on 100 validation queries whose top-5 results were rated against a written substitute rubric, by a rule I wrote first: the most substitutes in the top 5 wins, and anything within 1 point of the best counts as a tie that goes to the smallest λ. Substitutes rose from 63.2% at 0 to 68.8% at 0.15 (+5.6 [+3.2, +8.0]) and flattened beyond 0.20 (`results/substitute_policy.json`). The co-purchase proxy drops slightly, because I remove complements on purpose. I rejected a hard same-category filter (it fails the 15% without a category) and a learned category block (co-purchase would weight it for complements) |
| D25 | Test queries are fresh and family-disjoint, and labels are family-level | No test query shares a product group (name, family or photo) with any tuning, validation or early-check query. The early-check set is 2,000 queries I looked at during development, so I keep them out. Another size of a co-purchased item counts as the same correct answer. |
| D26 | A hand-labelled check of the ranking | The co-purchase proxy rewards complements, so I labelled results myself. I labelled 199 query-result pairs from 94 held-out test queries, in random order pooled across three systems, blind to the system, the rank and the category. Labels: substitute, partial, duplicate, not (definitions in `tools/human_check.py`). I called 96.8% of the served model's labelled results substitutes against 87.3% for text-only (+9.5 [+3.5, +16.3] points) and 96.6% for learned weights alone (`results/human_check/human_eval.md`) |
| D28 | Where a category is missing, wearer and product-type words in the names stand in for it, with a penalty of μ = 0.10 | Products without a category still name their wearer and type: men's vs women's, T-shirt vs jeans. Checked against rated pairs, a gender-word clash between two names meant "substitute" only 8% of the time, against 64% otherwise. For products with a category the category penalty already covers this (validation gain +0.0), so the rule applies only when either product's category is unknown. I chose μ on 60 products without a category (+5.0 [+1.3, +9.3]) and confirmed it on 100 fresh ones (+2.0 [+0.2, +4.2]), by a rule written first: at least 1 point with a CI above zero. Across all 242 such products in four samples the gain is +2.7 [+1.2, +4.4] (`results/name_policy.json`). The word lists are short and visible in `simsearch/data.py`. I rejected a hard filter, because a very close match should be able to stay |

## F. Approximate nearest neighbours (bonus)

D19. HNSW through FAISS, in two stages and opt-in. HNSW is from Malkov and Yashunin (IEEE TPAMI 2020,
arXiv:1603.09320). Because HNSW needs dense vectors, truncated SVD first compresses the sparse matrix to 256
dimensions; HNSW then proposes 200 candidates and the exact score re-ranks them. At 30,000 products it finds 96.2% of
the exact top 10 in 1.05 ms against 1.12 ms (`results/evaluation.md`). At this size exact search is
already about a millisecond, so it is the default and `SEARCH_MODE=hnsw` turns HNSW on. I chose FAISS over hnswlib because it offers other
index types behind the same interface, and over Annoy because Annoy has lower recall at the same speed on public
benchmarks. IVF-PQ is built for billion-scale catalogues. OPERATIONS.md describes when to switch.

## G. Service

| ID | Decision | Why, and what I rejected |
|---|---|---|
| D20 | Per-request weights (`?weights=`) and an LRU cache | The brief asks for configurable weighting. I validate the weights: an unknown block, a negative value, NaN, overflow or a repeated name returns 422. A shared Redis cache isn't needed for a query that takes about a millisecond |
| D21 | The HTTP contract | 404 for an unknown id (checked explicitly, so an internal error can never look like a missing product), 422 for invalid input, `num_similar` between 1 and 100, the index built at start-up rather than on the first request, and a JSON 500 with a request id for anything unexpected |
| D23 | A request with no usable signal is refused | `weights=image:1` on a product without a photo would rank by tie-breakers alone, so it returns 422 and names the attributes the product does have |
| D27 | The service shows which model it is serving | `config/manifest.json` hashes the data, embeddings, model config and serving code into a model version. It comes back as `X-Model-Version`, it is the Kubernetes image tag, and CI and a test check it. A mismatch or low photo coverage makes the instance degraded, and `/ready` then returns 503. `/metrics` exports latency, request counts, coverage, the live wrong-type share and catalogue exposure. MLflow would make sense with several models; for one, a manifest is enough |

## Next steps, in order

1. The online A/B test in OPERATIONS.md, whose click and add-to-cart logs give real substitution labels, and a
   multi-person labelling panel on the same labels.
2. Predict a product type from photo and name for the 15% of listings with no category.
3. Revisit the image adapter with substitution labels. In development I trained a linear InfoNCE adapter on
   co-purchase pairs, masking in-batch false negatives. It gained +0.023 on photos alone and 0.000 in the full
   model, so it waits for better labels.
4. Learning-to-rank on the per-attribute scores, once click data exists.
5. Per-customer indexes and a shared cache for many catalogues (OPERATIONS.md, "Scaling").
