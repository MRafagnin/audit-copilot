# ADR 002 — Hybrid retrieval (BM25 + dense, RRF k=60)

* **Status**: Accepted
* **Date**: 2026-05-22
* **Owner**: Matheus Rafagnin

## Context

The RAG corpus mixes two very different document classes:

* **AUASB ASA standards** — formal regulatory text, heavy on defined terms
  ("material misstatement", "tests of controls"). Exact-token recall matters.
* **ASX annual reports** — narrative prose with paraphrased disclosures.
  Semantic similarity matters.

Pure dense retrieval (sentence-transformers cosine) consistently missed
queries that named a standard by number (e.g. "ASA 240"). Pure BM25 missed
paraphrased queries against the annual report. The demo cannot afford
either failure mode in front of an interviewer.

## Decision

Retrieve top-`k` from **both** a BM25 index (`rank_bm25`) and a dense Chroma
collection (cosine over `all-MiniLM-L6-v2`), then fuse with
**reciprocal-rank fusion** (`k=60`), and return the top `RAG_TOP_K=8`
fused chunks. Refuse generation when the top fused score is below
`RAG_MIN_SCORE=0.35`.

## Alternatives considered

| Option | Why not |
|---|---|
| Dense only | Misses lexical hits on standard numbers / defined terms. |
| BM25 only | Misses paraphrases in the annual report. |
| Cross-encoder re-rank on top of dense | Adds a second model + latency for marginal gain on a 644-chunk corpus. Documented as a roadmap item. |
| ColBERT / late interaction | Operational complexity not justified at this scale. |
| Weighted score fusion | Requires per-corpus score calibration; RRF is rank-only and parameter-light. |

## Consequences

**Positive**

* Lexical and semantic queries both land on the right chunk.
* RRF needs no score calibration when the two retrievers move to different
  scales (e.g. Azure AI Search hybrid).
* The retrieval boundary is a single `HybridRetriever.search()` call — easy
  to mock in unit tests.

**Negative / accepted trade-offs**

* Two indexes to maintain. Both are rebuilt from the same chunks JSONL on
  `make bootstrap`, so drift is structural rather than operational.
* The min-score gate occasionally over-refuses on paraphrased queries. The
  golden-set evaluation tracks this (`citation_grounding_score`).

## Azure mapping

`Azure AI Search` with the **hybrid + semantic ranker** option replaces both
indexes in one service. The pipeline keeps the same `RAG_TOP_K` /
`RAG_MIN_SCORE` semantics; only the retriever implementation changes.
