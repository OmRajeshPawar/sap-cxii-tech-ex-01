"""Offline step 2: embed the cached product photos with Marqo-FashionSigLIP (GPU if available).

    python tools/embed_images.py   -> .cache/embeddings/fashion_siglip.npz

Writes L2-normalised float16 image embeddings. Heavy dependencies (torch, open_clip) live only in this offline
environment (requirements-offline.txt), never in the API image (D11). Why this encoder: DESIGN.md D10.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.data import load_raw  # noqa: E402

IMAGES = ROOT / ".cache" / "images"
OUT = ROOT / ".cache" / "embeddings"
MODEL, MODEL_ID, DESCRIPTION = "fashion_siglip", "hf-hub:Marqo/marqo-fashionSigLIP", "open_clip Marqo-FashionSigLIP"
BATCH = 128


def load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None   # missing or unreadable photo -> no image embedding (the D12 fallback)


@torch.inference_mode()
def main():
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    model.eval().to(device)
    ids = [p["uniq_id"] for p in load_raw()]

    start = time.perf_counter()
    feats, has_image = None, np.zeros(len(ids), dtype=bool)
    for lo in range(0, len(ids), BATCH):
        batch_ids = ids[lo:lo + BATCH]
        images = [load_image(IMAGES / f"{uid}.jpg") for uid in batch_ids]
        ok = [i for i, im in enumerate(images) if im is not None]
        if not ok:
            continue
        x = torch.stack([preprocess(images[i]) for i in ok]).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy()
        if feats is None:
            feats = np.zeros((len(ids), out.shape[1]), dtype=np.float16)
        feats[[lo + i for i in ok]] = out.astype(np.float16)
        has_image[[lo + i for i in ok]] = True
        if (lo // BATCH) % 10 == 0:
            print(f"{lo + len(batch_ids)}/{len(ids)} {time.perf_counter() - start:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{MODEL}.npz", ids=np.array(ids), embeddings=feats, has_image=has_image)
    meta = {"model": MODEL, "description": DESCRIPTION,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "dims": int(feats.shape[1]),
            "images_embedded": int(has_image.sum()), "image_seconds": round(time.perf_counter() - start, 1)}
    (OUT / f"{MODEL}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
