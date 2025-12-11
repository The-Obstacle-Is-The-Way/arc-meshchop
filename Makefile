.PHONY: help install install-dev sync lint format typecheck test test-fast test-cov clean pre-commit ci init

# Default Python version
PYTHON_VERSION := 3.10

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================
install:  ## Install production dependencies
	uv sync

install-dev:  ## Install all dependencies including dev
	uv sync --all-extras

sync:  ## Sync dependencies with lockfile
	uv sync --all-extras

# =============================================================================
# Code Quality
# =============================================================================
lint:  ## Run ruff linter
	uv run ruff check src tests

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check src tests --fix

format:  ## Run ruff formatter
	uv run ruff format src tests

format-check:  ## Check formatting without changes
	uv run ruff format src tests --check

typecheck:  ## Run mypy type checker
	uv run mypy src tests

quality:  ## Run all code quality checks (lint + format-check + typecheck)
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck

# =============================================================================
# Testing
# =============================================================================
test:  ## Run all tests
	uv run pytest

test-fast:  ## Run tests excluding slow markers
	uv run pytest -m "not slow"

test-cov:  ## Run tests with coverage report
	uv run pytest --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

test-parallel:  ## Run tests in parallel
	uv run pytest -n auto

test-verbose:  ## Run tests with verbose output
	uv run pytest -vvs

# =============================================================================
# Pre-commit
# =============================================================================
pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install

pre-commit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

# =============================================================================
# Development
# =============================================================================
clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# CI/CD Simulation
# =============================================================================
ci:  ## Run full CI pipeline locally
	$(MAKE) quality
	$(MAKE) test

# =============================================================================
# Project Initialization
# =============================================================================
init:  ## Initialize project (first-time setup)
	uv sync --all-extras
	uv run pre-commit install
	@echo "Project initialized successfully!"
