PYTHON ?= python3
NODE ?= node

.PHONY: check lint format test build run

check: lint test build

lint:
	@$(PYTHON) -m ruff --version >/dev/null 2>&1 || { \
		echo "make check needs ruff (pinned in requirements-dev.txt):"; \
		echo "  $(PYTHON) -m pip install -r requirements-dev.txt   (preferably inside a venv)"; \
		echo "or run with another interpreter: make check PYTHON=/path/to/venv/bin/python"; \
		exit 1; }
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest
	$(NODE) --test web/*.test.js

# Python is interpreted: validate bytecode compilation instead of producing a binary.
build:
	$(PYTHON) -m compileall -q app.py workspace.py run.py calibrate.py soccer_analyzer

run:
	$(PYTHON) -m streamlit run app.py
