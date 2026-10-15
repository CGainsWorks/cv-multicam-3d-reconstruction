# Multi-camera 3D reconstruction and 6-DoF pose.
#
# Three classes of target; the distinction matters when reading the README:
#
#   FAST      seconds to a few minutes, nothing to download.  Everything CI
#             runs, plus the end-to-end demo.
#               make test  make lint  make demo  make check
#
#   FULL      renders the full capture and reproduces every number the README
#             quotes.  `make data` is about twenty minutes and 900 MB on disk;
#             `make experiments` is about forty minutes after that.
#               make data  make calibrate  make reconstruct  make dense
#               make pose  make train  make ablate  make colmap
#
#   REPORT    turns finished runs into tables and figures.
#               make plots  make tables  make docs-check  make publish
#
# `make check` is exactly what CI does.

SHELL := /bin/sh
PYTHON ?= python
DATA ?= data/turntable
RESULTS ?= results
MODELS ?= models
STATION_STEP ?= 2
EPOCHS ?= 24

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Fast
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: check
check: lint test demo  ## Everything CI runs, no dataset needed

.PHONY: test
test:  ## Unit tests
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-fast
test-fast:  ## Unit tests without the slow whole-pipeline ones
	$(PYTHON) -m pytest tests/ -q -m "not slow"

.PHONY: lint
lint:  ## Formatting and obvious errors
	$(PYTHON) -m flake8 src/ tests/ scripts/ capture/

.PHONY: demo
demo:  ## Render a miniature capture and run the whole pipeline over it (~4 min)
	$(PYTHON) scripts/demo.py --out $(RESULTS)/demo.json

# ---------------------------------------------------------------------------
# Full
# ---------------------------------------------------------------------------

.PHONY: data
data:  ## Render the full capture (~20 min, ~900 MB under data/)
	$(PYTHON) scripts/make_dataset.py --out $(DATA)

.PHONY: calibrate
calibrate:  ## Per-camera intrinsics and rig extrinsics, with residuals
	$(PYTHON) scripts/calibrate.py --data $(DATA) --out $(RESULTS) \
		--refine-intrinsics-jointly

.PHONY: reconstruct
reconstruct:  ## Sparse reconstruction and bundle adjustment
	$(PYTHON) scripts/reconstruct.py --data $(DATA) --out $(RESULTS)

.PHONY: dense
dense:  ## Plane-sweep MVS, Open3D fusion and meshing
	$(PYTHON) scripts/dense.py --data $(DATA) --out $(RESULTS) \
		--station-step $(STATION_STEP)

.PHONY: pose
pose:  ## Classical 6-DoF pose on the held-out stations
	$(PYTHON) scripts/pose_eval.py --data $(DATA) --out $(RESULTS)

.PHONY: train
train:  ## Train the pose-refinement network
	$(PYTHON) scripts/train_refiner.py --data $(DATA) --out $(RESULTS) \
		--models $(MODELS) --epochs $(EPOCHS)

.PHONY: ablate
ablate:  ## The ablation: classical alone against classical plus the network
	$(PYTHON) scripts/ablate.py --data $(DATA) --out $(RESULTS)

.PHONY: colmap
colmap:  ## COLMAP as an independent reference for the sparse reconstruction
	$(PYTHON) scripts/colmap_baseline.py --data $(DATA) --out $(RESULTS)

.PHONY: experiments
experiments: calibrate reconstruct dense pose train ablate colmap  ## Everything the README quotes

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

.PHONY: plots
plots:  ## Regenerate every figure in docs/plots
	$(PYTHON) scripts/plots.py --results $(RESULTS) --out docs/plots

.PHONY: tables
tables:  ## Rewrite the README and docs/results.md tables from $(RESULTS)
	$(PYTHON) scripts/make_tables.py --results $(RESULTS)

.PHONY: docs-check
docs-check:  ## Do the documents still agree with the results they quote?
	$(PYTHON) scripts/check_docs.py --results $(RESULTS)

.PHONY: publish
publish:  ## Copy the result records into results/published, which is committed
	@mkdir -p $(RESULTS)/published
	@cp $(RESULTS)/*.json $(RESULTS)/published/ 2>/dev/null || true
	@ls $(RESULTS)/published/*.json | wc -l | xargs echo "records published:"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: setup
setup:  ## Virtual environment, dependencies, then the tests
	./scripts/setup.sh

.PHONY: clean
clean:  ## Remove generated results, figures and caches
	rm -rf $(RESULTS)/*.json $(RESULTS)/*.npz $(RESULTS)/*.ply $(RESULTS)/*.yaml
	rm -rf colmap_workspace
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

.PHONY: clean-data
clean-data:  ## Remove the rendered capture as well (it is reproducible)
	rm -rf $(DATA)

.PHONY: docker-build
docker-build:  ## Build the container
	docker build -f docker/Dockerfile -t cv-multicam-3d-reconstruction .

.PHONY: docker-demo
docker-demo:  ## Run the demo in the container, nothing to download
	docker compose -f docker/docker-compose.yml run --rm multicam
