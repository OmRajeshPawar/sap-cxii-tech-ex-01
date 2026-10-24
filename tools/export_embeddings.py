"""Offline step 3: export the photo embeddings as the small serving artifact data/image_embeddings.npz.

PCA to 64 dims happens here, once, so the API never needs the full embeddings or PyTorch (D11).
    python tools/export_embeddings.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.features import IMAGE_DIMS, IMAGE_FILE, reduce_embeddings  # noqa: E402

MODEL = "fashion_siglip"   # written by tools/embed_images.py


def main():
    data = np.load(ROOT / ".cache" / "embeddings" / f"{MODEL}.npz")
    has_image = data["has_image"]
    reduced = np.zeros((len(has_image), IMAGE_DIMS), dtype=np.float16)
    reduced[has_image] = reduce_embeddings(data["embeddings"][has_image].astype(np.float32)).astype(np.float16)
    np.savez_compressed(IMAGE_FILE, ids=data["ids"], embeddings=reduced, has_image=has_image)
    meta = json.loads((ROOT / ".cache" / "embeddings" / f"{MODEL}.json").read_text())
    meta.update({"exported_dims": IMAGE_DIMS, "pca": "sklearn PCA, random_state=0"})
    IMAGE_FILE.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {IMAGE_FILE} ({IMAGE_FILE.stat().st_size / 1e6:.1f} MB) from {MODEL}")


if __name__ == "__main__":
    main()
