# Makefile for Database Engine Agent

.PHONY: help install test run clean lint format check-style setup examples

# Default target
help:
	@echo "Database Engine Agent - Makefile Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo "  make setup        - Full setup (install + test)"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code with black"
	@echo "  make check-style  - Check code style"
	@echo ""
	@echo "Execution:"
	@echo "  make run          - Run agent with defaults"
	@echo "  make examples     - Run usage examples"
	@echo "  make demo         - Quick demo"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove generated files"
	@echo "  make clean-all    - Remove all artifacts"

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

setup: install test
	@echo "✓ Setup complete and tests passing"

# Testing
test:
	@echo "Running all tests..."
	python -m pytest test_agent.py -v

test-unit:
	@echo "Running unit tests..."
	python -m pytest test_agent.py -v -k "not Integration"

test-cov:
	@echo "Running tests with coverage..."
	python -m pytest test_agent.py --cov=agent --cov=config --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# Code quality
lint:
	@echo "Running flake8..."
	-flake8 agent.py config.py test_agent.py --max-line-length=100 --extend-ignore=E203,W503
	@echo "Running mypy..."
	-mypy agent.py config.py --ignore-missing-imports

format:
	@echo "Formatting code with black..."
	black agent.py config.py test_agent.py examples.py db_implementation.py --line-length=100

check-style:
	@echo "Checking code style..."
	black --check agent.py config.py test_agent.py --line-length=100

# Execution
run:
	@echo "Running Database Engine Agent..."
	python agent.py

demo:
	@echo "Running quick demo..."
	python agent.py --work-dir ./demo_output --checkpoint-interval 3

examples:
	@echo "Running examples..."
	python examples.py

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf *.pyc *.pyo
	rm -rf .coverage htmlcov
	@echo "✓ Cleanup complete"

clean-all: clean
	@echo "Removing all generated artifacts..."
	rm -rf output_* demo_output db_engine_output
	rm -rf checkpoints
	@echo "✓ Full cleanup complete"

# Development helpers
watch-test:
	@echo "Watching for changes and running tests..."
	while true; do \
		clear; \
		make test; \
		inotifywait -qre close_write .; \
	done

count-lines:
	@echo "Lines of code:"
	@wc -l *.py | sort -n

check: lint test
	@echo "✓ All checks passed"
