# ADR 001 — Local LLM via Ollama (qwen2.5:7b-instruct)

* **Status**: Accepted
* **Date**: 2026-05-22
* **Owner**: Matheus Rafagnin

## Context

AuditCopilot is a portfolio demo that must run **fully offline** on a Windows
laptop with Intune-managed restrictions (no Docker, no WSL2 — see ADR 005).
The demo path includes a live RAG Q&A and a fusion explainer that issues one
LLM call per flagged journal entry. Latency budget is "acceptable for a
2-minute live demo" — not production SLA.

Constraints:

* No outbound network for inference (data residency story).
* No GPU on the demo machine.
* Reproducible across two laptops (interview machine + backup).
* Must be swappable to a managed cloud model for the Azure roadmap.

## Decision

Use **Ollama** as the local model runtime, serving **qwen2.5:7b-instruct**
on `http://localhost:11434`. All LLM calls go through a single
`LLMClient` protocol (`src/llm/client.py`); the only implementation today is
`OllamaClient`. Azure OpenAI is the documented swap.

## Alternatives considered

| Option | Why not (now) |
|---|---|
| `llama3.1:8b-instruct` | Comparable quality, slightly slower CPU decode; kept as fallback if qwen output regresses. |
| `phi3:medium` | Smaller and faster but weaker on long-context grounded summarisation in early probes. |
| Direct `llama.cpp` Python bindings | More moving parts; loses Ollama's model registry + warm-load semantics. |
| Azure OpenAI from day one | Adds a network dependency to a demo that must run offline; also adds a billing surface to a portfolio project. |
| OpenAI API | Same as above; plus the data-residency story is weaker for an audit demo. |

## Consequences

**Positive**

* Zero network dependency for the demo; the laptop is the entire stack.
* `LLMClient` abstraction means the production swap is a new class + a
  config change, not a rewrite.
* Ollama's HTTP surface is small enough to mock cleanly in tests.

**Negative / accepted trade-offs**

* ~60 s per generation on CPU. Live demo uses the Streamlit "Explain" tab
  which sets expectations explicitly.
* Quality below GPT-4 class — mitigated by aggressive grounding +
  refusal-on-low-score guardrails (see ADR 004).

## Azure mapping

Replace `OllamaClient` with an `AzureOpenAIClient` implementing the same
protocol. Model: `gpt-4o-mini` for `/ask`, `gpt-4o` for `/explain`. Endpoint
+ key + deployment from environment, loaded by `src/core/config.py`. No
caller changes.
