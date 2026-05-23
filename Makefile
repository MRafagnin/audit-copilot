# AuditCopilot — common dev tasks (run via `uv run make <target>` or just `make <target>` once .venv is active)

.PHONY: help install bootstrap fetch-corpus gen-data run api ui test lint format typecheck eval clean

help:
	@echo "Targets:"
	@echo "  install      uv sync (installs runtime + dev dependencies)"
	@echo "  bootstrap    pull Ollama model, fetch corpus, generate GL data, build index"
	@echo "  run          start FastAPI + Streamlit side-by-side"
	@echo "  api          start FastAPI only"
	@echo "  ui           start Streamlit only"
	@echo "  test         pytest with coverage gate (>=80%)"
	@echo "  lint         ruff check + ruff format --check"
	@echo "  format       ruff format (writes changes)"
	@echo "  typecheck    mypy on src/"
	@echo "  eval         run golden-set + anomaly evaluation"
	@echo "  clean        remove caches and build artefacts"

install:
	uv sync --extra dev

bootstrap: install
	ollama pull qwen2.5:7b-instruct
	uv run python scripts/fetch_corpus.py
	uv run python scripts/gen_journal_entries.py --seed 42
	uv run python -m src.rag.indexer

fetch-corpus:
	uv run python scripts/fetch_corpus.py

gen-data:
	uv run python scripts/gen_journal_entries.py --seed 42

run:
	powershell -ExecutionPolicy Bypass -File scripts/run_dev.ps1

api:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run app.py

test:
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

eval:
	uv run pytest tests/eval -v
	uv run python -m src.anomaly.eval

clean:
	powershell -Command "Get-ChildItem -Recurse -Force -Include __pycache__,.pytest_cache,.mypy_cache,.ruff_cache,*.egg-info,htmlcov,.coverage | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
