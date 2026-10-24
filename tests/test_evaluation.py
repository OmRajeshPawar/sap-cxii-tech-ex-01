"""Weak labels, metrics, product families and fresh-query splits (simsearch/labels.py)."""
import numpy as np

from conftest import product
from simsearch.data import build_catalogue, family_key
from simsearch.labels import build_family_labels, build_labels, fresh_queries, ranking_metrics


def test_weak_labels_and_metrics():
    cat = build_catalogue([product("a", "red shoe", also="Blue Shoe | Not In Catalogue | red shoe"),
                           product("b", "blue shoe"), product("c", "green hat")])
    assert build_labels(cat) == {0: {"blue shoe"}}   # own name and unknown names are dropped
    m = ranking_metrics(["green hat", "blue shoe"], {"blue shoe"}, k=10)
    assert m["recall"] == 1.0 and m["hit_rate"] == 1.0 and m["mrr"] == 0.5 and 0 < m["ndcg"] < 1
    # standard recall divides by every relevant item, even beyond k; a repeated hit earns credit once
    assert ranking_metrics(["a"], {f"r{i}" for i in range(12)} | {"a"}, k=1)["recall"] == 1 / 13
    assert ranking_metrics(["a", "a"], {"a", "b"}, k=2)["recall"] == 0.5


def test_family_labels_and_fresh_queries_exclude_whole_product_groups():
    cat = build_catalogue([product("a", "red shoe (M)", also="Blue Shoe | red shoe (L)"), product("a2", "red shoe (L)"),
                           product("b", "blue shoe", also="red shoe (M)"), product("c", "green hat", also="blue shoe")])
    labels = build_family_labels(cat)
    assert labels[0] == {family_key("blue shoe")}   # its own family (another size) is never relevant
    groups = np.array([0, 0, 1, 2])
    assert fresh_queries(labels, groups, used=[1]) == [2, 3]   # a2 was used, so its whole group (a, a2) is out
