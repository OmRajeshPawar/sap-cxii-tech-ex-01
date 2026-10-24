"""HTTP contract, demo explanations and deployment consistency (app.py, demo.py, manifest)."""
from pathlib import Path

from fastapi.testclient import TestClient

from app import app
from simsearch.engine import calculate_similarity, find_similar_products, get_engine


def test_api_contract():
    with TestClient(app) as client:
        some_id = get_engine().cat.ids[0]
        ok = client.get("/find_similar_products", params={"product_id": some_id, "num_similar": 5})
        assert ok.status_code == 200 and len(ok.json()) == 5 and some_id not in ok.json()
        assert client.get("/find_similar_products", params={"product_id": "nope"}).status_code == 404
        for bad in ({"num_similar": 0}, {"num_similar": 101}, {"category_penalty": 2}, {"weights": "text=1"},
                    {"weights": "text:1,text:2"}, {"weights": "text:abc"}):
            assert client.get("/find_similar_products", params={"product_id": some_id, **bad}).status_code == 422
        assert client.get("/health").json() == {"status": "ok"}
        assert ok.headers["X-Model-Version"] and len(ok.headers["X-Request-ID"]) == 16
        assert client.get("/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"
        status = client.get("/api/status").json()
        assert status["products"] == 30000 and status["image_coverage"] > 0.99 and not status["degraded"]
        assert "similarity_requests_total" in client.get("/metrics").text
        no_image = next(i for i, has in zip(get_engine().cat.ids, get_engine().has_data[1]) if not has)
        assert client.get("/find_similar_products",
                          params={"product_id": no_image, "weights": "image:1"}).status_code == 422
        custom = client.get("/find_similar_products",
                            params={"product_id": some_id, "num_similar": 3, "weights": "image:0.7,text:0.3"})
        assert custom.status_code == 200 and len(custom.json()) == 3
        assert client.get("/ready").status_code == 200
        metrics = client.get("/metrics").text
        assert "similarity_latency_ms_count" in metrics and "# TYPE similarity_model_info gauge" in metrics
        live = client.get("/api/status").json()["live_quality"]
        assert 0 <= live["wrong_type_share"] <= 1 and live["catalogue_exposure"] > 0
        client.get("/find_similar_products", params={"product_id": some_id, "category_penalty": 0})
        assert client.get("/api/status").json()["live_quality"] == live   # experiments don't move the guardrail
        bad_id = client.get("/health", headers={"X-Request-ID": "x" * 500}).headers["X-Request-ID"]
        assert len(bad_id) == 16   # an implausible caller id is replaced, not echoed
        assert "similarity_wrong_type_share" in metrics and "similarity_catalogue_exposure" in metrics


def test_unexpected_error_returns_json_500_with_request_id(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure")
    with TestClient(app) as client:
        monkeypatch.setattr(get_engine(), "find_similar_products", boom)
        r = client.get("/find_similar_products", params={"product_id": get_engine().cat.ids[0]},
                       headers={"X-Request-ID": "trace-me"})
        assert r.status_code == 500 and r.json() == {"detail": "internal error", "request_id": "trace-me"}


def test_degraded_instance_stops_taking_traffic(monkeypatch):
    import app as service
    with TestClient(app) as client:
        monkeypatch.setitem(service.STATE, "manifest", {**service.STATE["manifest"], "verified": False,
                                                        "mismatched": ["model_config"]})
        ready = client.get("/ready")
        assert ready.status_code == 503 and "model_config" in str(ready.json()["reasons"])
        assert client.get("/health").status_code == 200   # liveness stays up: no restart loop


def test_demo_explanations_add_up_to_the_score():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/products", params={"limit": 1}).json()["items"]
        # a product with a known category, so every field (including same_type) is exercised
        data = client.get("/api/similar/2852dde03bc410bd2c8cf09db9ff674c", params={"num_similar": 4}).json()
        assert len(data["results"]) == 4
        for r in data["results"]:
            assert abs(sum(r["contributions"].values()) - r["score"]) < 1e-6
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)   # explanations match the ranking order


def test_deployment_tag_and_manifest_match_the_served_artifacts():
    import json
    import re
    from simsearch.manifest import MANIFEST, verify
    assert verify()["verified"]   # data, embeddings and model config hash to what the manifest says
    version = json.loads(MANIFEST.read_text())["model_version"]
    tag = re.search(r"image: product-similarity:(\w+)", (Path(__file__).parent.parent / "k8s.yaml").read_text()).group(1)
    assert tag == version   # the cluster deploys exactly the evaluated model


def test_brief_functions_agree_with_each_other():
    some_id = get_engine().cat.ids[0]
    results = find_similar_products(some_id, 5)
    scores = [calculate_similarity(some_id, other) for other in results]
    assert len(results) == 5 and scores == sorted(scores, reverse=True)   # ranked by exactly this score
