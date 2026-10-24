"""HTTP latency of the real service (D27), as distinct from the in-process engine time in evaluate.py.

Starts uvicorn (one worker) on a free local port, then calls GET /find_similar_products over keep-alive
connections: distinct products at concurrency 1 (cache misses), the same products again (cache hits), and
distinct products at concurrency 8. Client and server share one laptop, so these are local sanity numbers,
not a production SLO.

    python tools/http_bench.py                                   -> results/http_bench.json
    python tools/http_bench.py --url http://localhost:8000 --out results/http_bench_docker.json
                                 (an already running service, e.g. the Docker image; CI runs this on every push)
"""
import argparse
import http.client
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.data import load_catalogue  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(host, port, ids, workers):
    def worker(chunk):
        conn, out = http.client.HTTPConnection(host, port, timeout=30), []
        for pid in chunk:
            t = time.perf_counter()
            conn.request("GET", f"/find_similar_products?product_id={pid}&num_similar=10")
            resp = conn.getresponse()
            resp.read()
            out.append(((time.perf_counter() - t) * 1000, resp.status))
        conn.close()
        return out
    chunks = [ids[i::workers] for i in range(workers)]
    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        rows = [r for part in pool.map(worker, chunks) for r in part]
    wall = time.perf_counter() - start
    lat = np.array([ms for ms, _ in rows])
    return {"requests": len(rows), "concurrency": workers, "errors": sum(code != 200 for _, code in rows),
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)), "throughput_rps": len(rows) / wall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="benchmark this running service instead of starting one")
    parser.add_argument("--out", default="results/http_bench.json")
    args = parser.parse_args()
    if args.url:
        target = urlparse(args.url)
        host, port, server, client = target.hostname, target.port or 80, None, "same machine, keep-alive, Docker image"
    else:
        host, port, client = "127.0.0.1", free_port(), "same machine, keep-alive"
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(port), "--log-level",
                                   "warning"], cwd=ROOT, env={**os.environ, "SEARCH_MODE": "exact"})
    try:
        for _ in range(120):
            try:
                c = http.client.HTTPConnection(host, port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.5)
        ids = load_catalogue().ids
        rng = random.Random(0)
        misses, more = rng.sample(ids, 300), rng.sample(ids, 800)
        result = {"cache_miss_c1": run(host, port, misses, 1), "cache_hit_c1": run(host, port, misses, 1),
                  "cache_miss_c8": run(host, port, [i for i in more if i not in set(misses)][:400], 8),
                  "environment": {"python": platform.python_version(), "machine": platform.processor() or platform.machine(),
                                  "cpus": os.cpu_count(), "uvicorn_workers": 1, "client": client}}
    finally:
        if server:
            server.terminate()
    (ROOT / args.out).write_text(json.dumps(result, indent=2))
    for name, r in result.items():
        if name != "environment":
            print(f"{name:<14} p50 {r['p50_ms']:.2f} ms  p95 {r['p95_ms']:.2f} ms  p99 {r['p99_ms']:.2f} ms  "
                  f"{r['throughput_rps']:.0f} req/s  errors {r['errors']}")


if __name__ == "__main__":
    main()
