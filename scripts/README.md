# Scripts

Operational scripts invoked via `make` targets. Not part of the importable
`src/` package — these are entry points, not libraries.

- `fetch_corpus.py` — download AUASB ASA + IAASB ISA standards + the configured ASX annual report (Phase 1).
- `gen_journal_entries.py` — generate synthetic GL data with seeded anomalies (Phase 2).
- `run_dev.ps1` — start FastAPI + Streamlit side-by-side (PowerShell).
