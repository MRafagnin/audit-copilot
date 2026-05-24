# Scripts

Operational scripts invoked via `make` targets. Not part of the importable
`src/` package — these are entry points, not libraries.

- `fetch_corpus.py` — download AUASB ASA standards + the pre-indexed ASX annual report (Phase 1).
- `gen_journal_entries.py` — generate synthetic 50k-row GL with seeded anomalies and annual-report-style source-ref descriptions (Phase 2).
- `build_index.py` — chunk the corpus PDFs, embed with `all-MiniLM-L6-v2`, upsert into the embedded ChromaDB store. Called by `make bootstrap`.
- `train_anomaly.py` — train the ensemble (IsolationForest + autoencoder + KMeans), enrich cross-row feature flags, persist `data/flagged/top_k.csv` and `data/metrics/anomaly_eval.json`.
- `smoke_rag.py` — quick integration smoke test of the RAG pipeline (mocked LLM).
- `smoke_fusion.py` — quick integration smoke test of the fusion explainer end-to-end.
- `run_dev.ps1` — start FastAPI + Streamlit side-by-side (PowerShell).
