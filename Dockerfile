# Python 3.12: the pinned numpy / scikit-learn versions need 3.11+
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so code changes don't reinstall packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the service needs: code, the model (weights + policy + manifest of artifact hashes), data, the
# exported 64-d image embeddings (built offline by tools/, so this image needs no PyTorch) and the saved
# evaluation results the demo displays
COPY app.py demo.py ./
COPY simsearch/ simsearch/
COPY web/ web/
COPY config/model.json config/manifest.json config/
COPY results/evaluation.json results/demo_cases.json results/
COPY results/human_check/human_eval.json results/human_check/
COPY data/archive.zip data/image_embeddings.npz data/image_embeddings.json data/

# Run as an unprivileged user with a numeric UID, so Kubernetes can enforce runAsNonRoot
RUN useradd --create-home --uid 10001 appuser
USER 10001

EXPOSE 8000
ENV NAME=ProductSimilarityApp \
    PYTHONDONTWRITEBYTECODE=1 \
    SEARCH_MODE=exact

# One worker per container: a query is CPU-bound, so scale with replicas (k8s.yaml), not threads
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
