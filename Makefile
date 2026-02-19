.PHONY: install install-dev test lint fmt type-check serve clean docker-build docker-up

# ─── Setup ─────────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# ─── Quality ───────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x --ignore=tests/test_api

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

type-check:
	mypy src/code_indexer/

# ─── Dev server ────────────────────────────────────────────────────────────

serve:
	uvicorn code_indexer.api.app:app --reload --host 0.0.0.0 --port 8000

# ─── Docker ────────────────────────────────────────────────────────────────

docker-build:
	docker build -f docker/Dockerfile -t code-indexer:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down

# ─── Misc ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf dist/ build/ htmlcov/ .coverage .pytest_cache .ruff_cache .mypy_cache
