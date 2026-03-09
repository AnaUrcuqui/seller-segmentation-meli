.PHONY: install sync lint format typecheck test notebooks clean

install:
	uv sync --group dev
	uv run pre-commit install

sync:
	uv sync --group dev

data:
	uv run python scripts/download_data.py

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=seller_segmentation --cov-report=html

notebooks:
	uv run jupyter lab notebooks/

validate-nbs:
	uv run python scripts/validate_notebooks.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
