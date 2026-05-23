# GitHub Copilot Instructions — AuditCopilot

## Project Overview

**AuditCopilot** is a local-first, open-source AI assistant for Audit and Assurance work. It fuses two capabilities:

1. **RAG knowledge assistant** grounded on Australian auditing standards (**AUASB ASA**, with IAASB ISA kept as international reference) plus a real ASX annual report (default ticker: `WOW` — Woolworths Group). Answers cite source, section, and page. Refuses when retrieval confidence is low.2. **Journal-entry anomaly detection** on synthetic general-ledger data with seeded fraud patterns. Ensemble of `IsolationForest` + PyTorch autoencoder + KMeans clustering.

The **hero feature** is the fusion layer: when the anomaly engine flags a transaction, the RAG-grounded LLM produces a plain-English risk narrative citing **ASA 240** / fraud-risk indicators (ISA 240 equivalence noted where relevant).

Portfolio project for the Deloitte **AI Engineer | ML, GenAI, LLM, Python, Azure** role in Audit and Assurance. See `PLAN.md` for the full design and `PREREQUISITES.md` for setup.

### Primary goals

1. **Grounded, safe LLM output** — every answer cites sources; guardrails block prompt injection, PII leakage, and ungrounded responses.
2. **End-to-end ML lifecycle** — ingest, train, evaluate, serve, monitor.
3. **Production-quality code** — typed, tested (pytest ≥80% on changed files), linted, CI-gated.
4. **Azure-ready architecture** — each local component maps to a managed Azure service. Documented in README, not built.

### Important instructions

- Don't be sycophantic. If the idea is bad, say so.
- Don't touch config files (`pyproject.toml`, `.env`, `Makefile`, CI YAML, `.github/`) without asking.
- Run tests and lint before committing. Every time.
- When unsure, stop and ask. Don't guess.
- This is a **portfolio demo** — clarity, defensibility, and the live-demo path beat feature breadth.

---

## Tech Stack

### Core

- **Python 3.11** (pinned — best compatibility for PyTorch / sentence-transformers / ChromaDB as of mid-2026)
- **`uv`** for dependency and venv management (no `pip`, no `poetry`)
- **Native runtime** — no Docker locally (Intune blocks WSL2 on the dev machine; see ADR 005)

### Key libraries

```python
# RAG
chromadb              # embedded PersistentClient — data/chroma/
sentence-transformers # all-MiniLM-L6-v2 embeddings
rank_bm25             # lexical retrieval for hybrid fusion
pypdf                 # PDF parsing for standards corpus

# LLM
httpx                 # Ollama HTTP client (qwen2.5:7b-instruct on localhost:11434)

# ML / anomaly detection
numpy, pandas
scikit-learn          # IsolationForest, KMeans, metrics
torch                 # autoencoder

# Serving
fastapi, uvicorn      # backend API
streamlit             # demo UI

# Data sourcing
requests              # AUASB / IAASB / ASX PDF fetch
faker                 # synthetic GL data

# Tooling
pytest, pytest-cov
ruff, black, mypy
pre-commit
python-dotenv
```

### External services (local)

- **Ollama** — Windows service on `http://localhost:11434`, serving `qwen2.5:7b-instruct`. Production swap documented: `llama3.1:8b` locally, `gpt-4o-mini` / `gpt-4o` on Azure OpenAI.
- **ChromaDB** — embedded mode, persistent store at `data/chroma/`. No server process.

---

## Architecture

### Layered layout

```
┌────────────────────────────────────────────────────┐
│  app.py (Streamlit)  +  src/api/main.py (FastAPI) │  ← Presentation
├────────────────────────────────────────────────────┤
│  src/fusion/explain.py                             │  ← Hero: anomaly + RAG
├────────────────────────────────────────────────────┤
│  src/rag/        │  src/anomaly/                   │  ← Domain logic
│    pipeline.py   │    detectors.py                 │
│    guardrails.py │    features.py                  │
│    indexer.py    │    eval.py                      │
│    ingest.py     │                                 │
├────────────────────────────────────────────────────┤
│  src/llm/client.py (LLMClient interface)           │  ← Integration
│  src/core/{config,constants,logging_config}.py     │
├────────────────────────────────────────────────────┤
│  scripts/{fetch_corpus,gen_journal_entries,        │  ← Data + ops
│           run_dev.ps1}.py                          │
└────────────────────────────────────────────────────┘
```

### Design principles

1. **Separation of concerns** — RAG, anomaly, fusion, LLM client, and API layers do not import across boundaries except through defined interfaces.
2. **LLM provider is abstracted** — `src/llm/client.py` defines `LLMClient`; only one implementation (`OllamaClient`) exists today. Azure OpenAI swap is a single new class.
3. **Config from environment** — `.env` (never committed) loaded via `python-dotenv` in `src/core/config.py`. No hardcoded secrets, no hardcoded paths.
4. **Deterministic where possible** — synthetic data, seeded models, fixed embedding model versions. Demos must reproduce.
5. **Fail closed on safety** — guardrails refuse rather than hallucinate. Low retrieval confidence returns "insufficient grounding — manual review required".
6. **Observable** — structured logging at DEBUG/INFO/WARNING/ERROR; a `/health` endpoint; metrics JSON persisted per evaluation run.

---

## Coding Standards

### Style

- **PEP 8** enforced by `ruff` and `black` (line length 100).
- **`mypy` strict on `src/`** — every function signature typed.
- **`ruff` rules**: at minimum `E`, `F`, `I` (imports), `B` (bugbear), `UP` (pyupgrade), `SIM` (simplify).

### Imports

```python
# Standard library
import os
import logging
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import pandas as pd
from fastapi import FastAPI

# Local
from src.core.config import settings
from src.llm.client import LLMClient
```

### Type hints

Required on every function and method signature.

```python
def retrieve(
    query: str,
    *,
    k: int = 5,
    min_score: float = 0.35,
) -> list[RetrievedChunk]:
    ...
```

### Docstrings

Use **Google-style docstrings**. Every public function, class, and method must have one. Private helpers (`_prefix`) need a docstring only when logic is non-obvious.

Rules:

1. **Args match the signature exactly** — same names, same order.
2. **Document every parameter** — never partial.
3. **Returns describes structure** — for dicts list the keys; for tuples describe each element. Not just "dict" or "tuple".
4. **Raises** when a named exception propagates to callers.
5. **No dynamic data in the summary line** — it is a static description, not a log message.
6. **Class docstrings** describe purpose and key attributes — not just "wrapper for X".

Example:

```python
def explain_anomaly(
    transaction: JournalEntry,
    retriever: HybridRetriever,
    llm: LLMClient,
    *,
    min_grounding_score: float = 0.4,
) -> ExplainResult:
    """Produce a grounded risk narrative for a flagged journal entry.

    Retrieves the top-k audit-standard chunks relevant to the transaction's
    features (round amount, weekend posting, unusual user-account pair, etc.)
    and asks the LLM to explain why it is risky, citing the retrieved chunks.

    Args:
        transaction: The flagged journal entry, including anomaly score and
            triggered feature flags.
        retriever: Hybrid (BM25 + dense) retriever over the audit-standards
            corpus.
        llm: LLM client used to generate the narrative.
        min_grounding_score: If the top retrieval score is below this
            threshold, the explainer refuses rather than hallucinates.

    Returns:
        ExplainResult with fields:
            - narrative (str): Plain-English risk explanation, or refusal text.
            - citations (list[Citation]): Source, section, page for each cited
              chunk. Empty when the explainer refused.
            - grounded (bool): False when retrieval confidence was below
              `min_grounding_score`.

    Raises:
        LLMClientError: If the LLM call fails after retries.
    """
```

### Error handling

- Catch the specific exception. Never bare `except`.
- Log with structured context, then re-raise (or return a typed error result at API boundaries).
- Boundary validation only — trust internal callers, validate user input and external API responses.

```python
try:
    chunks = retriever.search(query, k=k)
except RetrievalError as exc:
    logger.error("retrieval failed", extra={"query_len": len(query), "k": k})
    raise
```

### Logging

- Get the module logger: `logger = logging.getLogger(__name__)`.
- **Never call `logging.basicConfig` in a module.** Logging is configured once in `src/core/logging_config.py` and invoked from `src/api/main.py` and `app.py` startup.
- Message strings are **static, lowercase noun-phrase labels** — no f-strings, no dynamic data inlined. Dynamic data goes in `extra={}`.
- Levels: `DEBUG` for diagnostics, `INFO` for progress, `WARNING` for recoverable issues, `ERROR` for failures.
- No emojis in log messages.

```python
# Good
logger.info("rag pipeline started")
logger.info("retrieval completed", extra={"k": k, "top_score": scores[0]})
logger.warning("grounding score below threshold", extra={"top_score": scores[0]})
logger.error("llm call failed", extra={"model": model_name})

# Wrong
logger.info(f"Retrieved {len(chunks)} chunks for query '{query}'")
logger.warning("⚠️  low score")
```

- Default log format is JSON; switch to text locally via `LOG_FORMAT=text` env var.
- Never log secrets, the configured `HTTP_USER_AGENT`, raw user queries containing PII, or full prompts. Scrub at the logging boundary.

---

## Naming Conventions

- **snake_case** for variables, functions, modules.
- **PascalCase** for classes (`HybridRetriever`, `OllamaClient`, `JournalEntry`, `ExplainResult`).
- **UPPERCASE** for constants (`CHUNK_SIZE`, `EMBEDDING_MODEL`, `OLLAMA_BASE_URL`).
- Verb-noun for actions (`extract_iocs`, `score_transaction`, `build_index`).
- Predicates use `is_`, `has_`, `check_`, `validate_`.
- Module name matches the primary class or capability (`indexer.py` → `Indexer`).

---

## Configuration

### Environment variables (`.env`)

```bash
# LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b-instruct
LLM_TIMEOUT_SECONDS=120

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector store
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=audit_corpus

# RAG
RAG_TOP_K=8
RAG_MIN_SCORE=0.35

# HTTP (polite scraping — identify the caller when fetching AUASB / ASX PDFs)
HTTP_USER_AGENT=Matheus Rafagnin matheus.rafagnin@gmail.com

# Corpus
DEMO_TICKER=WOW

# Anomaly
ANOMALY_SEED=42
GL_ROW_COUNT=50000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

Loaded once in `src/core/config.py` into a pydantic `Settings` object. Modules import `from src.core.config import settings` — never call `os.getenv` outside `config.py`.

### Constants

Static values that are not user-tunable live in `src/core/constants.py`:

```python
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
EMBED_DIM = 384  # all-MiniLM-L6-v2

# Standards URLs (public)
AUASB_STANDARDS_INDEX_URL = "https://..."
IAASB_HANDBOOK_URL = "https://..."
```

---

## RAG Pipeline Conventions

### Chunking

- Recursive character splitter, **512 tokens** with **64-token overlap**.
- Preserve source metadata on every chunk: `source` (AUASB / IAASB / ASX-`<ticker>`), `section`, `page`, `chunk_id`.

### Retrieval

- Hybrid: dense (`all-MiniLM-L6-v2` via Chroma) **+** BM25 (`rank_bm25`).
- Fuse with **reciprocal-rank fusion** (`k=60`).
- Return the top `RAG_TOP_K` after fusion.

### Generation

- System prompt is **locked** in `src/rag/prompts.py`. User input is templated, never concatenated as a string.
- Output schema is structured: `{answer: str, citations: list[{source, section, page, chunk_id}]}`. Parse and validate before returning to the API layer.
- Refuse when the top fused score is below `RAG_MIN_SCORE`.

### Guardrails (`src/rag/guardrails.py`)

- Prompt-injection check on user input (regex heuristics + LLM-as-judge).
- PII scrub on user input (emails, SSNs, credit cards). Optional Microsoft Presidio path.
- Citation enforcement: refusal text when grounding is insufficient.
- Every refusal is logged with reason at `WARNING`.

---

## Anomaly Detection Conventions

- Synthetic GL data is generated by `scripts/gen_journal_entries.py` with `--seed 42`. Schema: `date, account, debit, credit, user, posting_ts, description, is_anomaly` (ground-truth label).
- Features (`src/anomaly/features.py`): temporal (weekend, after-hours), user-account frequency, Benford's-law first digit, amount z-score, description-embedding centroid distance.
- Three models in `src/anomaly/detectors.py`:
  - `IsolationForest` (sklearn) — baseline.
  - `Autoencoder` (PyTorch, 3-layer MLP) — reconstruction error as score.
  - `KMeans` — clusters risk patterns; cluster ID is a feature for the explainer.
- Ensemble: normalize each score to `[0,1]`, weighted average, weights calibrated on a held-out split.
- Evaluation (`src/anomaly/eval.py`): precision@k, ROC-AUC, PR-AUC. Persist to `data/metrics/anomaly_eval.json`.

---

## Testing Standards

### Layout

```
tests/
  rag/
  anomaly/
  fusion/
  api/
  eval/
    golden_set.py   # 10–15 Q&A pairs for faithfulness / citation / refusal
conftest.py         # shared fixtures
```

### Rules

- **Hermetic** — no network. Mock the Ollama HTTP client and the Chroma client at their boundary.
- **Deterministic** — set seeds for NumPy, PyTorch, and Python's `random`.
- **Coverage** — `pytest --cov=src --cov-fail-under=80`. CI fails below 80% on changed files.
- One file per module under test (`tests/rag/test_pipeline.py` mirrors `src/rag/pipeline.py`).
- Use `pytest.mark.parametrize` for input-table tests.
- `tests/eval/golden_set.py` runs as a smoke test in CI: faithfulness ≥ 0.85, citation accuracy checks, refusal-behavior assertions.

### Mocking pattern

```python
from unittest.mock import Mock

def test_pipeline_refuses_on_low_score(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_retriever = Mock()
    mock_retriever.search.return_value = [RetrievedChunk(score=0.1, ...)]
    mock_llm = Mock()

    result = answer_question("what is ISA 240?", mock_retriever, mock_llm)

    assert result.refused is True
    assert mock_llm.complete.call_count == 0  # never called when refusing
```

---

## API Conventions (FastAPI)

- Endpoints: `POST /ask`, `POST /scan`, `GET /explain/{tx_id}`, `GET /health`.
- Request and response models are pydantic; OpenAPI is auto-generated.
- Every handler logs a request-scoped event at `INFO` on entry and exit.
- Validation errors return 422 with the pydantic detail; safety refusals return 200 with a structured `{refused: true, reason: ...}` body — they are not errors.

---

## Security

1. **Never commit secrets** — `.env` is in `.gitignore`. `.env.example` is the only checked-in template.
2. **Scrub PII in logs** — emails, SSNs, credit cards. Never log full prompts or raw user input.
3. **Validate inputs at the API boundary** — pydantic models with strict types and length caps on free-text fields.
4. **Prompt injection** — system prompt is locked; user input is templated; injection-detection heuristics run before the LLM call.
5. **HTTPS only** for external fetches (AUASB, IAASB, ASX investor-relations pages).
6. **Dependencies** — track via `uv.lock`; review `uv pip list --outdated` and known-vulnerable advisories before each release.
7. **No code execution from retrieved content** — all retrieved chunks are text; never `eval`, `exec`, or template-render them.

---

## Git Workflow

### Branch naming

```
feat/<short-description>
fix/<short-description>
refactor/<short-description>
docs/<short-description>
chore/<short-description>
```

### Commit messages (Conventional Commits)

```
feat(rag): add reciprocal-rank fusion for hybrid retrieval
fix(anomaly): correct Benford first-digit feature on negative amounts
refactor(llm): extract LLMClient interface for Azure OpenAI swap
docs(readme): add Azure deployment mapping table
test(rag): add prompt-injection probe cases
chore(ci): bump uv to 0.11.x in workflow
```

### Pre-commit / CI gates

- `ruff check`, `ruff format --check`, `mypy src/`, `pytest --cov=src --cov-fail-under=80`.
- Conventional Commit lint on PR titles.
- Golden-set evaluation runs in CI as a smoke test.

---

## When Writing New Code

### Always consider

1. **Is it on the demo path?** If not, document in roadmap instead of building.
2. **Where does it fail safely?** Refuse over hallucinate; structured error over crash.
3. **Can it be unit-tested with mocks?** If not, redesign for testability.
4. **Is the LLM call necessary?** If a deterministic check works, skip the LLM.
5. **Does it have an Azure mapping?** Components without a clean Azure path are a smell.

### Code review checklist

- [ ] Type hints on every signature
- [ ] Google-style docstrings (Args match signature, Returns describes structure, Raises when applicable)
- [ ] Errors caught specifically and logged with `extra={}`
- [ ] No secrets, no `os.getenv` outside `src/core/config.py`
- [ ] Constants from `src/core/constants.py`, settings from `src/core/config.py`
- [ ] Unit tests hermetic, with seeds set
- [ ] Coverage ≥ 80% on changed files
- [ ] No new dependencies without a one-line justification in the PR

---

## Quick Reference

### Essential files

```
PLAN.md                          # Full project plan and decisions
PREREQUISITES.md                 # Local setup checklist (complete)
.env                             # Secrets (gitignored)
.env.example                     # Template

src/core/config.py               # Settings (pydantic)
src/core/constants.py            # Static constants
src/core/logging_config.py       # Logging setup

src/llm/client.py                # LLMClient interface + OllamaClient

src/rag/ingest.py                # Corpus → chunks
src/rag/indexer.py               # Chunks → ChromaDB
src/rag/pipeline.py              # Retrieval + generation
src/rag/guardrails.py            # Safety layer
src/rag/prompts.py               # Locked system prompts

src/anomaly/features.py
src/anomaly/detectors.py
src/anomaly/eval.py

src/fusion/explain.py            # Hero: anomaly + RAG

src/api/main.py                  # FastAPI
app.py                           # Streamlit

scripts/fetch_corpus.py
scripts/gen_journal_entries.py
scripts/run_dev.ps1
```

### Make targets

```
make bootstrap   # ollama pull, fetch corpus, generate GL data, build index
make run         # FastAPI (uvicorn) + Streamlit side-by-side
make test        # pytest with coverage gate
make lint        # ruff + mypy
make eval        # golden-set + anomaly metrics
```

### Default ports

```
FastAPI    : http://localhost:8000
Streamlit  : http://localhost:8501
Ollama     : http://localhost:11434  (Windows service)
```

### Performance defaults

```
RAG_TOP_K            = 8
RAG_MIN_SCORE        = 0.35
CHUNK_SIZE_TOKENS    = 512
CHUNK_OVERLAP_TOKENS = 64
GL_ROW_COUNT         = 50000
ANOMALY_SEED         = 42
```

---

**Last Updated**: May 22, 2026
**Version**: 0.1.0
**Owner**: Matheus Rafagnin
