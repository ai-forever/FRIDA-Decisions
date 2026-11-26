RUFF ?= uvx ruff
MYPY ?= uvx mypy

.PHONY: format format-check lint lint-fix mypy

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

lint:
	$(RUFF) check .

lint-fix:
	$(RUFF) check --fix .

mypy:
	$(MYPY) --config-file pyproject.toml
