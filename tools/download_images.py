"""Offline step 1: download each product's first 'medium' image once into .cache/images/.

The embedding step reads these cached files, so a re-run embeds exactly the same inputs.
    python tools/download_images.py          # ~750 MB, ~3-5 min with 32 threads; safe to re-run (skips cached)
"""
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.data import load_raw  # noqa: E402

CACHE = ROOT / ".cache" / "images"
WORKERS, TIMEOUT, RETRIES = 32, 15, 2


def first_url(product) -> str:
    return str(product.get("medium") or product.get("large") or "").split("|")[0].strip()


def fetch(job) -> bool:
    uid, url = job
    target = CACHE / f"{uid}.jpg"
    if target.exists() and target.stat().st_size > 0:
        return True
    if not url.startswith("http"):
        return False
    for _ in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                target.write_bytes(r.read())
            return True
        except Exception:
            continue
    return False


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(p["uniq_id"], first_url(p)) for p in load_raw()]
    with ThreadPoolExecutor(WORKERS) as pool:
        ok = 0
        for i, success in enumerate(pool.map(fetch, jobs), 1):
            ok += success
            if i % 2000 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} processed, {ok} images cached", flush=True)
    print(f"done: {ok}/{len(jobs)} images in {CACHE}")


if __name__ == "__main__":
    main()
