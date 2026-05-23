# AuditCopilot

> Local-first, open-source AI assistant for Audit and Assurance. RAG over Australian auditing standards (**AUASB ASA**, with IAASB ISA as international reference) and a curated allowlist of ASX annual reports (default: **Woolworths Group**, `WOW`; 10 tickers ingestable on demand from the UI), plus a journal-entry anomaly-detection engine that explains each flagged transaction with cited audit-standard references.

**Status:** Phase 5 — end-to-end demo live (RAG `/ask`, anomaly `/scan`, fusion `/explain`, Streamlit UI). See [PLAN.md](PLAN.md) for the full roadmap and [docs/adr/](docs/adr/) for design decisions.

## What it does

1. **Ask the auditor assistant.** Hybrid RAG (BM25 + dense + reciprocal-rank fusion) over AUASB ASA and the selected ASX annual report (default `WOW`; pick another from the sidebar dropdown). Every answer cites source, section, and page. Refuses when retrieval confidence is below threshold.
2. **Scan journal entries.** Synthetic general-ledger data with seeded fraud patterns is scored by an ensemble of `IsolationForest` + PyTorch autoencoder + KMeans.
3. **Explain anomalies.** The hero feature — flagged transactions get a plain-English risk narrative grounded in **ASA 240** / fraud-risk indicators, with inline citations.

## Architecture

```mermaid
flowchart LR
    subgraph UI
        ST[Streamlit app.py]
    end
    subgraph API[FastAPI]
        ASK[/POST /ask/]
        SCAN[/POST /scan/]
        EXP[/GET /explain/:tx_id/]
        HEALTH[/GET /health/]
    end
    subgraph RAG[src/rag]
        GUARD[guardrails]
        PIPE[pipeline]
        IDX[ChromaIndex + BM25]
    end
    subgraph ANOM[src/anomaly]
        FEAT[features]
        ENS[IsolationForest + AE + KMeans]
    end
    subgraph FUSION[src/fusion]
        EXPL[AnomalyExplainer]
    end
    LLM[(Ollama qwen2.5:7b)]
    CHROMA[(ChromaDB data/chroma)]
    GL[(data/flagged/top_k.csv)]

    ST --> ASK
    ST --> SCAN
    ST --> EXP
    ASK --> GUARD --> PIPE --> IDX --> CHROMA
    PIPE --> LLM
    SCAN --> GL
    EXP --> EXPL --> PIPE
    EXPL --> LLM
```

## Tech stack

Python 3.11, `uv`, FastAPI + Streamlit, ChromaDB (embedded), sentence-transformers (`all-MiniLM-L6-v2`), Ollama (`qwen2.5:7b-instruct`), scikit-learn, PyTorch, pytest. No Docker locally (see [ADR 005](docs/adr/005-native-runtime.md)).

## Quickstart

```powershell
# Prereqs: see PREREQUISITES.md
git clone https://github.com/MRafagnin/audit-copilot.git
cd audit-copilot
Copy-Item .env.example .env   # then edit HTTP_USER_AGENT
uv sync --extra dev
uv run make bootstrap         # pulls model, fetches corpus, generates GL data, builds index
uv run make run               # FastAPI on :8000, Streamlit on :8501
```

## Project layout

```
src/
  core/          # config, constants, logging
  llm/           # LLMClient interface + OllamaClient
  rag/           # ingest, indexer, pipeline, guardrails, prompts
  anomaly/       # features, detectors, eval
  fusion/        # explain (anomaly + RAG)
  api/           # FastAPI
app.py           # Streamlit
scripts/         # fetch_corpus, gen_journal_entries, run_dev.ps1
tests/           # mirrors src/, plus tests/eval/ (golden set)
docs/adr/        # architecture decision records
```

## Make targets

| Target       | What it does                                         |
| ------------ | ---------------------------------------------------- |
| `install`    | `uv sync --extra dev`                                |
| `bootstrap`  | pull model, fetch corpus, gen GL data, build index   |
| `run`        | FastAPI + Streamlit side-by-side                     |
| `test`       | pytest with coverage gate (≥ 80%)                    |
| `lint`       | ruff check + format check                            |
| `typecheck`  | mypy on `src/`                                       |
| `eval`       | golden-set + anomaly metrics                         |

## AI safety inventory

Every LLM call is wrapped in a fail-closed guardrail stack (full rationale in [ADR 004](docs/adr/004-guardrails.md)):

| Layer                              | Where                                  | Failure mode addressed                  |
| ---------------------------------- | -------------------------------------- | --------------------------------------- |
| Input length cap (2 000 chars)     | `src/api/schemas.py`                   | Prompt-injection surface area           |
| Prompt-injection regex             | `src/rag/guardrails.py`                | Jailbreak / instruction override        |
| PII scrub (email, TFN, card)       | `src/rag/guardrails.py`                | Sensitive data leakage to the LLM       |
| Min retrieval score (refusal)      | `src/rag/pipeline.py` (`RAG_MIN_SCORE`)| Ungrounded / hallucinated answers       |
| Locked system prompt               | `src/rag/prompts.py`                   | Prompt concatenation attacks            |
| Citation enforcement               | `src/rag/pipeline.py`                  | Fabricated citations                    |
| Structured logging (no secrets)    | `src/core/logging_config.py`           | Secret / PII leakage via logs           |

Refusals return HTTP 200 with `{"refused": true, "reason": "..."}` — they are valid responses, not errors.

## Evaluation results

Persisted to `data/metrics/` and regenerated by `make eval`.

**Anomaly detection** (`anomaly_eval.json`, 50 000-row synthetic GL, seed 42, weights `iso=0.25 / ae=1.0 / kmeans=0.0`)

| Metric         | Value |
| -------------- | ----- |
| Precision@100  | 0.98  |
| ROC-AUC        | 0.884 |
| PR-AUC         | 0.646 |

**RAG golden set** (`golden_eval.json`, 8 grounded + 3 injection-refusal entries)

| Metric                      | Value | Threshold |
| --------------------------- | ----- | --------- |
| Citation grounding (top-k)  | 0.875 | ≥ 0.85    |
| Injection refusal accuracy  | 1.00  | = 1.00    |

## Add a new ticker

The ASX allowlist lives in [src/rag/registry.py](src/rag/registry.py) as `ASX_ANNUAL_REPORTS`. Each entry is an `AnnualReport(ticker, name, url, fy_label)` pointing at a publicly hosted annual-report PDF.

To add a company:

1. Append a new entry to `ASX_ANNUAL_REPORTS` with the ticker (uppercase), company name, direct PDF URL, and the financial-year label (e.g. `"FY24"`).
2. Restart the API (`uv run make run`) — the sidebar dropdown picks it up automatically.
3. In the Streamlit UI, select the ticker. If it is not yet indexed, click **Ingest \<TICKER\>**; the synchronous `POST /companies/{ticker}/ingest` endpoint downloads the PDF, chunks it, and upserts into ChromaDB under `source=ASX-<TICKER>`. Retrieval for that company filters on `{"source": {"$in": ["AUASB", "ASX-<TICKER>"]}}`.

Ingest is idempotent: re-clicking a ticker that already has chunks is a no-op. Only `WOW` ships pre-indexed via `make bootstrap`; other tickers are fetched on demand.

## Why no Docker locally?

The development laptop is Intune-managed and blocks WSL2 / Docker Desktop. The stack runs as native Windows processes (Python via `uv`, Ollama as a Windows service, ChromaDB embedded). Containerisation happens at the **deploy** boundary, not the dev boundary — see [ADR 005](docs/adr/005-native-runtime.md).

## Azure mapping

| Local component               | Azure equivalent                              |
| ----------------------------- | --------------------------------------------- |
| Ollama (`qwen2.5:7b`)         | Azure OpenAI (`gpt-4o-mini` / `gpt-4o`)       |
| ChromaDB (embedded)           | Azure AI Search (vector + hybrid + semantic)  |
| Regex injection check         | Azure AI Content Safety — Prompt Shields      |
| Regex PII scrub               | Azure AI Language — PII detection             |
| FastAPI (`uv run`)            | Azure Container Apps                          |
| Streamlit                     | Azure App Service                             |
| Model artefacts               | Azure ML model registry + endpoint            |
| `data/chroma` / `data/raw`    | Azure Data Lake Storage Gen2                  |
| `.env` secrets                | Azure Key Vault                               |
| GitHub Actions                | Azure DevOps Pipelines                        |
| Structured JSON logs          | Azure Monitor / Log Analytics                 |

## Architecture decisions

* [ADR 001 — Local LLM via Ollama](docs/adr/001-local-llm.md)
* [ADR 002 — Hybrid retrieval (BM25 + dense, RRF)](docs/adr/002-hybrid-retrieval.md)
* [ADR 003 — Ensemble anomaly detection](docs/adr/003-ensemble-anomaly.md)
* [ADR 004 — Guardrails: fail closed over hallucinate](docs/adr/004-guardrails.md)
* [ADR 005 — Native Windows runtime (no Docker locally)](docs/adr/005-native-runtime.md)

## Roadmap

* **Cross-encoder re-ranking** on top of hybrid retrieval to lift the golden-set grounding score above 0.95.
* **LLM-as-judge** injection probe as a second guardrail layer (latency-tolerant deployments only).
* **Knowledge-graph layer** over named auditing concepts (ASA → assertion → procedure) for explainer prompt augmentation.
* **Fine-tuning** a small open model on AUASB-style narrative summaries.
* **Azure IaC** (Bicep) for the full deployment mapping above.

## License

MIT. See [LICENSE](LICENSE) once added.
