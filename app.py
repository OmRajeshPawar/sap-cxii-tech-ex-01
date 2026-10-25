"""FastAPI service: GET /find_similar_products (D24-D25), plus health, readiness, status and metrics (D27)."""
import logging
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from demo import router as demo_router
from simsearch.engine import get_engine, parse_weights
from simsearch.features import BLOCKS
from simsearch.manifest import verify

# Process-wide counters. Endpoints run in a thread pool, so updates take the lock.
STATE = {"started": time.time(), "manifest": {}, "requests": {}, "latency_ms": deque(maxlen=2000),
         "latency_sum_ms": 0.0, "latency_count": 0,
         # live quality signals (D27): results whose known product type differs from the query's, and exposure
         "typed_results": 0, "wrong_type_results": 0, "exposed": set()}
LOCK = threading.Lock()
log = logging.getLogger("similarity")
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")   # a caller's id is echoed and logged only if it looks like one


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_engine()                 # build the index once, before the first request
    STATE["manifest"] = verify()  # which artifacts and code were loaded, and do they match config/manifest.json?
    yield


app = FastAPI(title="Product Similarity Service", lifespan=lifespan)


@app.middleware("http")
async def measure(request: Request, call_next):
    """Request id, per-route request counts by status and server-side latency, for /api/status and /metrics.
    An unexpected exception becomes a JSON 500 that carries the request id, which is also logged with the trace."""
    request_id = request.headers.get("x-request-id", "")
    if not REQUEST_ID.fullmatch(request_id):
        request_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error, request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse(status_code=500, content={"detail": "internal error", "request_id": request_id})
    ms = (time.perf_counter() - start) * 1000
    route = request.scope.get("route")
    key = (route.path if route else "other", response.status_code)
    with LOCK:
        STATE["requests"][key] = STATE["requests"].get(key, 0) + 1
        if key[0] == "/find_similar_products":
            STATE["latency_ms"].append(ms)
            STATE["latency_sum_ms"] += ms
            STATE["latency_count"] += 1
    response.headers["X-Model-Version"] = STATE["manifest"].get("model_version", "unknown")
    response.headers["Server-Timing"] = f"app;dur={ms:.2f}"
    response.headers["X-Request-ID"] = request_id
    return response


def record_results(product_id: str, ids: List[str]) -> None:
    """Live signals from what was actually served. Wrong-type share: of results where both leaf categories are
    known, how many differ from the query's; a regression signal for the category rule, compared with its own
    baseline. Exposure: share of the catalogue recommended since start-up; it only grows within a process, so a
    trend needs a fixed window (OPERATIONS.md)."""
    engine = get_engine()
    code, idx = engine.category_code, [engine.index[i] for i in ids]
    q = code[engine.index[product_id]]
    typed = [code[i] for i in idx if code[i] >= 0] if q >= 0 else []
    with LOCK:
        STATE["typed_results"] += len(typed)
        STATE["wrong_type_results"] += sum(c != q for c in typed)
        STATE["exposed"].update(idx)


def status() -> dict:
    """What this instance is serving and how healthy it is (also shown in the demo's operator view)."""
    engine = get_engine()
    coverage = float(engine.has_data[BLOCKS.index("image")].mean())
    manifest = STATE["manifest"]
    reasons = [f"image embeddings cover {coverage:.0%} of products (expected ~99.5%)"] if coverage < 0.9 else []
    if manifest and not manifest.get("verified"):   # anything short of a full verification takes the pod out of rotation
        if manifest.get("mismatched") or manifest.get("missing"):
            reasons.append(f"artifacts differ from config/manifest.json: {manifest.get('mismatched') or manifest.get('missing')}")
        if manifest.get("dependencies"):
            reasons.append(f"installed libraries differ from requirements.txt: {manifest['dependencies']}")
        if not (manifest.get("mismatched") or manifest.get("missing") or manifest.get("dependencies")):
            reasons.append("config/manifest.json is missing or lists no artifact hashes")
    with LOCK:
        lat = np.array(STATE["latency_ms"]) if STATE["latency_ms"] else None
        requests = dict(STATE["requests"])
    return {"model_version": manifest.get("model_version"), "manifest_verified": manifest.get("verified"),
            "degraded": bool(reasons), "degraded_reasons": reasons,
            "search_mode": "hnsw" if engine.ann is not None else "exact",
            "products": len(engine.cat), "image_coverage": coverage,
            "weights": engine.weights, "category_penalty": engine.category_penalty,
            "cache": engine._cached.cache_info()._asdict(),
            "live_quality": live_quality(len(engine.cat)),
            "uptime_s": round(time.time() - STATE["started"]),
            "requests": {f"{path} {code}": n for (path, code), n in sorted(requests.items())},
            "server_errors": sum(n for (_, code), n in requests.items() if code >= 500),
            "latency_ms_recent": None if lat is None else {"n": int(len(lat)), "p50": float(np.percentile(lat, 50)),
                                                           "p95": float(np.percentile(lat, 95))}}


def live_quality(n_products: int) -> dict:
    with LOCK:
        typed, wrong, exposed = STATE["typed_results"], STATE["wrong_type_results"], len(STATE["exposed"])
    return {"wrong_type_share": wrong / typed if typed else None, "typed_results": typed,
            "catalogue_exposure": exposed / n_products}


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up and answering. Deliberately trivial, so a bad artifact set cannot crash-loop it."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: 503 while degraded (artifacts differ from the manifest, or photo coverage is low), so Kubernetes
    stops routing traffic to an instance that is not serving the evaluated model."""
    s = status()
    if s["degraded"]:
        return JSONResponse(status_code=503, content={"status": "degraded", "reasons": s["degraded_reasons"]})
    return {"status": "ready", "model_version": s["model_version"]}


@app.get("/api/status")
def api_status() -> dict:
    return status()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text format. Counters and _sum/_count aggregate across replicas; the quantiles are per instance."""
    s = status()
    with LOCK:
        requests, total_ms, count = dict(STATE["requests"]), STATE["latency_sum_ms"], STATE["latency_count"]
    lines = ["# TYPE similarity_requests_total counter"]
    lines += [f'similarity_requests_total{{route="{path}",code="{code}"}} {n}' for (path, code), n in sorted(requests.items())]
    lines += ["# TYPE similarity_latency_ms summary"]
    if s["latency_ms_recent"]:
        lines += [f'similarity_latency_ms{{quantile="0.5"}} {s["latency_ms_recent"]["p50"]:.3f}',
                  f'similarity_latency_ms{{quantile="0.95"}} {s["latency_ms_recent"]["p95"]:.3f}']
    lines += [f"similarity_latency_ms_sum {total_ms:.3f}", f"similarity_latency_ms_count {count}",
              "# TYPE similarity_degraded gauge", f"similarity_degraded {int(s['degraded'])}",
              "# TYPE similarity_image_coverage gauge", f"similarity_image_coverage {s['image_coverage']:.4f}",
              "# TYPE similarity_model_info gauge", f'similarity_model_info{{version="{s["model_version"]}"}} 1',
              "# TYPE similarity_catalogue_exposure gauge",
              f"similarity_catalogue_exposure {s['live_quality']['catalogue_exposure']:.4f}"]
    if s["live_quality"]["wrong_type_share"] is not None:
        lines += ["# TYPE similarity_wrong_type_share gauge",
                  f"similarity_wrong_type_share {s['live_quality']['wrong_type_share']:.4f}"]
    return "\n".join(lines) + "\n"


def require_product(product_id: str) -> None:
    """404 for an unknown id, checked explicitly so that an internal KeyError can never look like 'not found'."""
    if product_id not in get_engine().index:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")


@app.get("/find_similar_products")
def get_similar_products(
    product_id: str = Query(..., min_length=1),
    num_similar: int = Query(10, ge=1, description="how many similar products to return (default 10)"),
    weights: Optional[str] = Query(None, description="optional per-request block weights, e.g. image:0.6,text:0.4"),
    category_penalty: Optional[float] = Query(None, ge=0, le=1,
                                              description="optional override of the category penalty; 0 turns off both "
                                                          "product-type rules (category and name words)"),
) -> List[str]:
    require_product(product_id)
    try:
        ids = get_engine().find_similar_products(product_id, num_similar, parse_weights(weights), category_penalty)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    if not weights and category_penalty is None:   # only the served configuration feeds the live signals
        record_results(product_id, ids)
    return ids


app.include_router(demo_router)   # demo UI and explanations (demo.py)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
