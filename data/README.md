# Data directory

Everything under here is **gitignored** — corpus PDFs, generated GL data, the
ChromaDB persistent store, and evaluation metrics are all reproduced locally
via `uv run make bootstrap` (corpus + GL + index) and `uv run python
scripts/train_anomaly.py` (flagged top-k + metrics).

Structure (populated by bootstrap; per-ticker ASX folders appear on first
ingest via the UI or `POST /companies/{ticker}/ingest`):

```
data/
  corpus/
    auasb/           # AUASB ASA standards (PDF)
    asx/
      wow/           # Woolworths annual report (PDF) — pre-indexed by bootstrap
      cba/           # other allowlist tickers populate on demand
      tls/
      ...            # csl, nab, anz, rio, mqg
  chunks/            # chunks.jsonl — chunked corpus payload for the indexer
  chroma/            # embedded ChromaDB persistent store
  gl/                # journal_entries.csv — synthetic 50k-row general ledger
  flagged/           # top_k.csv — top-N anomaly rows with feature flags
  metrics/           # anomaly_eval.json, golden_eval.json
```
