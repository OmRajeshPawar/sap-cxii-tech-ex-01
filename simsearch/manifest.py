"""Model manifest (D27): which artifacts and code make up the served model, so a running instance can say
exactly what it is serving and whether that matches what was evaluated.

    python -m simsearch.manifest   -> config/manifest.json and the k8s.yaml image tag (run after any model or code change)

The model version is a short hash over the artifact hashes: change the data, the image embeddings, the
weights/policy file, the serving code or a pinned library version and the version changes. At start-up the service
re-hashes the files it loaded, checks the installed libraries against requirements.txt, and compares both with the
manifest (`verify`); any difference is reported as degraded and /ready returns 503.
"""
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "config" / "manifest.json"
ARTIFACTS = {"data": ROOT / "data" / "archive.zip",
             "image_embeddings": ROOT / "data" / "image_embeddings.npz",
             "model_config": ROOT / "config" / "model.json"}


def sha256(path: Path) -> str:
    """Content hash. Text files are hashed with LF line endings, so a Windows checkout (CRLF) and a Linux
    checkout of the same commit get the same hash (.gitattributes stores text as LF)."""
    if path.suffix in {".json", ".txt", ".md", ".py", ".html"}:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The serving code is part of the model: changing the scoring or the API must change the version (and the image tag).
# requirements.txt pins every library the service imports, so a dependency upgrade is a new version too.
SERVING_CODE = ["app.py", "demo.py", "web/index.html", "simsearch/*.py", "requirements.txt"]
REQUIREMENTS = ROOT / "requirements.txt"


def dependency_mismatches() -> list:
    """Installed library versions that differ from the pins in requirements.txt (empty when the environment matches).
    The pins are hashed into the version, so this check ties the running environment to the version as well."""
    from importlib.metadata import PackageNotFoundError, version
    out = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        spec = line.split("#")[0].strip()
        if "==" not in spec:
            continue
        name, pinned = (part.strip() for part in spec.split("=="))
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "missing"
        if installed != pinned:
            out.append(f"{name} {installed} (pinned {pinned})")
    return out


def code_hash() -> str:
    """One hash over every file the service runs, line endings normalised, in a fixed order."""
    digest = hashlib.sha256()
    for path in sorted(p for pattern in SERVING_CODE for p in ROOT.glob(pattern)):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def current_hashes() -> Dict[str, str]:
    hashes = {name: sha256(path) if path.exists() else "missing" for name, path in ARTIFACTS.items()}
    return {**hashes, "serving_code": code_hash()}


def version_of(hashes: Dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]


def build() -> dict:
    hashes = current_hashes()
    model = json.loads(ARTIFACTS["model_config"].read_text())
    encoder = ROOT / "data" / "image_embeddings.json"
    manifest = {
        "model_version": version_of(hashes),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": hashes,
        "image_encoder": json.loads(encoder.read_text()) if encoder.exists() else None,
        "weights": model["weights"],
        "policy": model.get("policy", {}),
        "evaluation": "results/evaluation.json (co-purchase proxy, held-out test) and results/human_check/human_eval.json (hand labels)",
        "splits": "simsearch/labels.py query_splits(seed=0) + fresh_queries; see DESIGN.md D17, D20",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    # Deploy exactly this version: point the Kubernetes image tag at it (tests/test_service.py checks they match)
    k8s = ROOT / "k8s.yaml"
    k8s.write_text(re.sub(r"(image: product-similarity:)\w+", rf"\g<1>{manifest['model_version']}",
                          k8s.read_text(encoding="utf-8")), encoding="utf-8")
    return manifest


def verify() -> dict:
    """What this process loaded vs the manifest: {model_version, verified, mismatched, missing}."""
    hashes = current_hashes()
    expected = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    want = expected.get("artifacts", {})
    dependencies = dependency_mismatches()
    return {"model_version": version_of(hashes), "manifest_version": expected.get("model_version"),
            "verified": bool(want) and want == hashes and not dependencies, "dependencies": dependencies,
            "mismatched": sorted(k for k in hashes if want and want.get(k) != hashes[k]),
            "missing": sorted(k for k, h in hashes.items() if h == "missing")}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
