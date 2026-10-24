# Serving (light) -------------------------------------------------------------
install:            ## API + evaluation + test dependencies
	pip install -r requirements-dev.txt

test:               ## unit + API tests
	python -m pytest -q

run:                ## start the API and demo on :8000
	uvicorn app:app --port 8000

analyse:            ## first look at the raw data -> analysis/data_report.md and analysis/figures/
	python analysis/explore_data.py

evaluate:           ## proxy metrics on fresh test queries, baselines, ablation, exact vs HNSW -> results/evaluation.*
	python evaluate.py

bench:              ## HTTP latency of the real service -> results/http_bench.json
	python tools/http_bench.py

manifest:           ## hash the served artifacts -> config/manifest.json (run after any model change)
	python -m simsearch.manifest

tune:               ## re-learn block weights on tuning queries -> config/model.json
	python tune.py

docker:             ## build and run the container, tagged with the model version
	docker build -t product-similarity:$$(python -c "import json;print(json.load(open('config/manifest.json'))['model_version'])") . && \
	docker run -p 8000:8000 product-similarity:$$(python -c "import json;print(json.load(open('config/manifest.json'))['model_version'])")

human-check:        ## score my hand labels -> results/human_check/human_eval.*
	python tools/human_check.py

static:             ## the served model's answers for every product in one page -> docs/index.html (checks 300 against the engine)
	python tools/export_static_demo.py

cases:              ## pick the demo's guided-tour cases -> results/demo_cases.json
	python tools/demo_cases.py

# Offline photo pipeline (heavy, optional: the exported artifact is already committed) -------
embed-install:      ## torch / open_clip in a separate environment
	pip install -r requirements-offline.txt

images:             ## download all product photos once (~750 MB, cached)
	python tools/download_images.py

embed:              ## embed the photos with Marqo-FashionSigLIP -> .cache/embeddings/
	python tools/embed_images.py

export:             ## PCA to 64-d -> data/image_embeddings.npz (the file the API reads)
	python tools/export_embeddings.py

.PHONY: install test run analyse evaluate bench manifest tune docker human-check cases static embed-install images embed export
