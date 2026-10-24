I labelled 199 result pairs from 94 held-out test queries, blind to the system, the rank and the category. Labels: substitute, partial, duplicate, not (definitions in `tools/human_check.py`).

| System | Labelled results | Called a substitute | Substitute or partial |
|---|---|---|---|
| Text only (TF-IDF) | 118 | 87.3% | 94.9% |
| Learned weights only | 119 | 96.6% | 99.2% |
| Served | 126 | 96.8% | 99.2% |

Served minus text-only: +9.5 [+3.5, +16.3] points. Served minus learned weights only: +0.2 [-3.0, +3.3] points. 95% bootstrap intervals over queries.
