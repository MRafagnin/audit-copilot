# Data directory

Everything under here is **gitignored** — corpus PDFs, generated GL data, the
ChromaDB persistent store, and evaluation metrics are all reproduced locally
via `uv run make bootstrap`.

Structure (created on first run):

```
data/
  corpus/auasb/      # AUASB ASA standards (PDF)
  corpus/iaasb/      # IAASB ISA standards (PDF, international reference)
  corpus/asx/        # ASX annual report for DEMO_TICKER
  chroma/            # embedded ChromaDB persistent store
  raw/               # raw downloads cache
  metrics/           # evaluation JSON for README badges
```
