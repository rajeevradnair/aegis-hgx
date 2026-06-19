.PHONY: install test lint format check clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[ml, serving]"

test:
	python -m pytest

lint:
	python -m ruff check . --fix

format:
	python -m ruff format .

check: lint test

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +