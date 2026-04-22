# ─────────────────────────────────────────────────────────────────────────
#   Volatility Forecasting API — developer commands
#   Every day-to-day action has a single-word verb. No surprises.
# ─────────────────────────────────────────────────────────────────────────
.PHONY: help setup install lint fmt test cov run docker-build docker-run \
        artifacts baselines stats clean

PYTHON ?= python
IMAGE  ?= volatility-api:latest
PORT   ?= 8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────
setup:  ## Create venv and install dev dependencies
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt

install:  ## Install runtime dependencies only
	pip install -r requirements.txt

# ── Quality ──────────────────────────────────────────────────────────────
lint:  ## Lint (ruff)
	ruff check app scripts tests

fmt:  ## Auto-format (ruff)
	ruff format app scripts tests
	ruff check --fix app scripts tests

# ── Tests ────────────────────────────────────────────────────────────────
test:  ## Run unit tests
	pytest -q

cov:  ## Run unit tests with coverage report
	pytest --cov=app --cov=scripts --cov-report=term-missing

# ── Offline artifacts (run once after training) ──────────────────────────
baselines:  ## Compute Diebold-Mariano comparison → app/baselines.json
	$(PYTHON) -m scripts.compute_baselines \
	    --results results.pkl --out app/baselines.json

stats:  ## Compute training-distribution stats → app/training_stats.json
	$(PYTHON) -m scripts.compute_training_stats \
	    --features features.pkl --scalers app/scalers.joblib \
	    --out app/training_stats.json

artifacts: baselines stats  ## Generate all offline artifacts in one shot

# ── Run ──────────────────────────────────────────────────────────────────
run:  ## Start the API locally (reload on code change)
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:  ## Build the production image
	docker build -t $(IMAGE) .

docker-run:  ## Run the production image
	docker run --rm -p $(PORT):8000 $(IMAGE)

# ── Housekeeping ─────────────────────────────────────────────────────────
clean:  ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
