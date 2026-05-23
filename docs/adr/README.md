# Architecture Decision Records

Short, immutable records of significant decisions. Format: context → decision → consequences.

Planned ADRs for the v0.1 release:

- `001-local-llm-choice.md` — `qwen2.5:7b-instruct` for the demo, `llama3.1:8b` documented as the conservative swap, Azure OpenAI as the production target.
- `002-hybrid-retrieval.md` — BM25 + dense, fused with reciprocal-rank fusion.
- `003-ensemble-anomaly.md` — IsolationForest + PyTorch autoencoder + KMeans, weighted ensemble.
- `004-guardrail-strategy.md` — locked system prompt, templated user input, PII scrub, refusal on low grounding.
- `005-native-runtime-no-docker.md` — `uv run` + embedded ChromaDB instead of `docker compose`, driven by Intune WSL2 policy on the dev machine.

Each ADR is written when the corresponding decision is implemented, not before.
