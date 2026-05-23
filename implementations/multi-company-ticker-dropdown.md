# Plan: Multi-company annual reports via ticker dropdown

**Date**: 2026-05-22
**Status**: Planned (not yet implemented)
**Branch:** `feature/multi-company-ticker-dropdown` (off `main`)
**Files to Change**:

- `scripts/fetch_corpus.py` — extend `ASX_ANNUAL_REPORTS` allowlist (10 tickers); expose `_download_one` for reuse.
- `src/core/config.py` — `demo_ticker` becomes a default selection, not a hard wire.
- `src/rag/registry.py` *(new)* — company list + `indexed` probe.
- `src/rag/ingest_company.py` *(new)* — orchestrates fetch → chunk → upsert.
- `src/rag/_http.py` *(new)* — shared PDF downloader extracted from `fetch_corpus.py`.
- `src/rag/indexer.py` — `ChromaIndex.query(where=...)` passthrough.
- `src/rag/pipeline.py` — `company` parameter; per-source BM25 filtering; 3-tuple `bm25_documents` shape.
- `src/api/state.py` — emit `source` from BM25 loader; expose `reset_caches` on the ingest path.
- `src/api/schemas.py` — `company` on `AskRequest`; new `CompanyOut` / `CompaniesResponse` / `IngestResponse`.
- `src/api/main.py` — `GET /companies`, `POST /companies/{ticker}/ingest`; thread `company` into `/ask` and `/explain/{tx_id}`.
- `app.py` — sidebar ticker selectbox + ingest-on-select flow; templated example questions.
- `tests/rag/test_indexer.py`, `tests/rag/test_pipeline.py`, `tests/rag/test_ingest_company.py` *(new)*, `tests/api/test_main.py` — coverage for filter, idempotent ingest, endpoints.
- `README.md` — "Add a new ticker" section; hero copy + screenshot caption.

```powershell
git checkout -b feature/multi-company-ticker-dropdown
```

Extend AuditCopilot from the hardcoded single-company (WOW) corpus to a
curated list of ASX tickers. User picks a ticker in the UI; the system
fetches and indexes that annual report on demand (cached on disk), and
retrieval mixes fixed AUASB standards with **only** the selected company's
chunks.

## Approach

- **Curated allowlist** of ASX tickers in `scripts/fetch_corpus.py`
  (`ASX_ANNUAL_REPORTS: dict[str, AnnualReport]`), seeded with **10** ASX 50
  names. Each entry carries `ticker`, `name`, `url`, `fy_label`. URLs
  reviewed manually — keeps the audit-trail posture intact.
- **Single Chroma collection**, filtered by `source` metadata. Standards
  keep source `AUASB`; each company keeps `ASX-<TICKER>` (already the
  convention). Retrieval filters dense + BM25 to
  `{"AUASB", "ASX-<selected>"}` so other tickers' chunks never bleed in.
- **On-demand ingest**: when a user picks a ticker not yet indexed, the API
  downloads the PDF, chunks it, embeds + upserts into Chroma, appends rows
  to `data/chunks/chunks.jsonl`, and invalidates the BM25 cache. Subsequent
  selections hit cache (file + Chroma).
- **UI**: replace the implicit `ASX-WOW` pill with a ticker dropdown in
  the sidebar; selection is per-session and threaded through `/ask` and
  `/explain/{tx_id}`.

## Resolved decisions

- **Allowlist size = 10.** Required: `WOW`, `CBA`, `BHP`, `WES`, `TLS`
  (from `PREREQUISITES.md` §10). Remaining 5 slots filled from ASX 50
  names with publicly hosted FY25 annual-report PDFs — working candidates:
  `CSL` (healthcare), `NAB` and `ANZ` (banks — diversify vs. CBA),
  `RIO` (mining, complement to BHP), `MQG` (financial services). Verify
  each URL during Phase 1; swap if a PDF 404s.
- **No on-disk eviction.** Once ingested, a ticker's PDF and chunks stay
  until manually deleted.
- **Anomaly side: defer.** GL stays synthetic and company-agnostic. The
  selected ticker only influences RAG retrieval (not `/scan` or which
  transactions are flagged). Surface as a roadmap note in README later if
  needed.
- **Curated allowlist over auto-discovery.** Matches the existing
  audit-trail rationale in `fetch_corpus.py`; avoids brittle IR-page
  scraping.
- **Single Chroma collection with metadata filter** over per-company
  collections. Fewer moving parts, BM25 stays a single in-memory model,
  embedder loads once.
- **Synchronous ingest endpoint** for v1. Background-task queue
  (FastAPI `BackgroundTasks` / Celery) deferred — adds complexity the demo
  doesn't need.
- **`demo_ticker` becomes a default**, not a hard pin. Existing `.env`
  keeps working.

## Steps

### Phase 1 — Corpus registry + filtered retrieval (foundational)

1. In `scripts/fetch_corpus.py`, refactor `ASX_ANNUAL_REPORTS` from
   `dict[str, str]` to `dict[str, AnnualReport]` with `name`, `url`,
   `fy_label`; seed the 10 entries. Update `_output_path`,
   `download_corpus`, and the CLI `--ticker` flag to use the new struct.
2. Add `src/rag/registry.py`: `list_companies()` returning the allowlist
   as `[CompanyInfo(ticker, name, fy_label, indexed: bool)]`. `indexed`
   is computed from presence of `data/corpus/asx/<ticker>/*.pdf`.
3. In `src/rag/indexer.py` `ChromaIndex.query`, accept optional
   `where: dict | None` and pass through to Chroma (`$in` filter on
   `source`). Default behaviour unchanged when `where is None`.
4. In `src/rag/pipeline.py`:
   - Extend `RagPipeline.retrieve` / `answer` with optional
     `company: str | None`. When set, build
     `sources = ["AUASB", f"ASX-{company}"]`, pass as
     `where={"source": {"$in": sources}}` to `index.query`, and filter
     BM25 ranking by the same set (use a chunk-id → source map built
     when BM25 is constructed).
   - Change `bm25_documents` signature to include source:
     `list[tuple[str, str, str]]` → `(chunk_id, source, text)`. Update
     callers and tests.
5. Update `src/api/state.py` `get_bm25_documents()` to read `source`
   from the JSONL and emit 3-tuples. Wire `reset_caches()` into the
   ingest path.

### Phase 2 — On-demand ingest endpoint

6. Add `src/rag/ingest_company.py` with
   `ingest_company(ticker) -> IngestResult`:
   - Look up entry in registry; reject unknown tickers.
   - If PDF missing on disk, download (reuse `_download_one` from
     `fetch_corpus.py`, extracted into a small `_http.py` helper to
     avoid a circular import).
   - `chunk_pdf(..., source=f"ASX-{ticker}")`, append to
     `data/chunks/chunks.jsonl`, upsert to `ChromaIndex`.
   - Idempotent: if Chroma already has chunks for that source (count
     via a `where` filter), skip.
7. Add FastAPI endpoints in `src/api/main.py`:
   - `GET /companies` → list registry + indexed flag.
   - `POST /companies/{ticker}/ingest` → runs `ingest_company`; returns
     `{ticker, chunks_added, took_ms}`. Synchronous; invalidates the
     BM25 cache afterwards.
8. Add `company: str | None` to `AskRequest` and `ExplainResponse`
   plumbing. `/ask` and `/explain/{tx_id}` thread it into pipeline calls.

### Phase 3 — Streamlit UI (parallel with Phase 2 once endpoints stubbed)

9. Sidebar: replace the hardcoded `ASX-WOW` pill with a `st.selectbox`
   populated from `GET /companies`. Persist choice in
   `st.session_state["company"]`.
10. When the user picks an un-indexed ticker, show a spinner and call
    `POST /companies/{ticker}/ingest`. On success, refresh the dropdown
    indexed-state and update the hero pill (`ASX-<TICKER>`).
11. Update example questions to be templated on the selected company
    name (e.g., "What does the {name} annual report disclose about …").
12. Thread `company` into `/ask` and `/explain/{tx_id}` calls.

### Phase 4 — Tests + docs

13. Tests:
    - `tests/rag/test_pipeline.py`: BM25 + dense filter by source set;
      refusal when company has no chunks.
    - `tests/rag/test_indexer.py`: `query(where=...)` passes through.
    - `tests/rag/test_ingest_company.py`: mock HTTP + Chroma;
      idempotency.
    - `tests/api/test_main.py`: `/companies`,
      `/companies/{ticker}/ingest` (mock downloader + embedder),
      `/ask` honours `company`.
14. README: short "Add a new ticker" section pointing at the allowlist;
    update the Hero copy and screenshot caption.

## Relevant files

- `scripts/fetch_corpus.py` — extend `ASX_ANNUAL_REPORTS` allowlist;
  expose `_download_one` for reuse.
- `src/core/config.py` — `demo_ticker` becomes a **default** selection,
  not a hard wire. No schema change.
- `src/rag/registry.py` *(new)* — company list + `indexed` probe.
- `src/rag/ingest_company.py` *(new)* — orchestrates fetch → chunk →
  upsert.
- `src/rag/indexer.py` — `ChromaIndex.query(where=...)`.
- `src/rag/pipeline.py` — `company` parameter; per-source BM25
  filtering; 3-tuple `bm25_documents` shape.
- `src/api/state.py` — emit `source` from BM25 loader; `reset_caches`.
- `src/api/schemas.py` — add `company` to `AskRequest`,
  `CompanyOut` / `CompaniesResponse`, `IngestResponse`.
- `src/api/main.py` — `/companies`, `/companies/{ticker}/ingest`;
  thread `company` into `/ask`, `/explain/{tx_id}`.
- `app.py` — sidebar ticker selectbox + ingest-on-select flow.
- `data/corpus/asx/<ticker>/*.pdf`, `data/chunks/chunks.jsonl` — storage
  layout unchanged (already per-ticker subfolder).

## Verification

1. `uv run pytest -q` green; coverage ≥ 80% on changed files
   (`pytest --cov=src --cov-fail-under=80`).
2. `uv run ruff check && uv run ruff format --check && uv run mypy src/`.
3. Manual: `uv run python scripts/fetch_corpus.py` still produces the
   WOW PDF in the same path. Existing index continues to answer
   "ASA 240" questions.
4. Manual end-to-end: start API + Streamlit, pick an un-indexed ticker
   (e.g. `CBA`), observe spinner → ingest completes → `/ask` returns a
   CBA-grounded answer with `ASX-CBA` citations and **no** WOW
   citations.
5. Manual isolation check: with `CBA` selected, ask a WOW-specific
   question ("What did Woolworths report for Australian Food?") —
   expect refusal or a standards-only answer, never WOW figures.
6. `GET /companies` reflects `indexed=true` after the ingest call.

## Excluded scope

- User-supplied PDF upload (file input). Can be added later as a second
  path alongside the ticker dropdown.
- Auto-detecting latest FY for each ticker. The allowlist pins a
  specific FY per entry; bumping FY is a manual PR.
- Multi-company simultaneous retrieval (cross-company comparison
  questions).
- Tying GL / anomaly detection to the selected ticker (deferred — see
  Resolved decisions).
