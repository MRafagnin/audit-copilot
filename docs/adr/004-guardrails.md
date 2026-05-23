# ADR 004 — Guardrails: fail closed over hallucinate

* **Status**: Accepted
* **Date**: 2026-05-22
* **Owner**: Matheus Rafagnin

## Context

An audit assistant that confidently invents a citation is worse than one
that refuses. The audience for this demo (audit partners + AI engineers)
will probe for hallucination and prompt injection. The system must refuse
visibly when it should, and never produce ungrounded text dressed up with
fake citations.

## Decision

A layered, **fail-closed** guardrail stack on both sides of every LLM call:

1. **Input — length cap.** The API rejects free-text fields >2 000
   characters (pydantic `max_length`). Bounds the prompt-injection surface.
2. **Input — prompt-injection check.** Regex heuristics in
   `src/rag/guardrails.py` detect canonical jailbreak patterns ("ignore
   previous instructions", "disregard the system prompt", "you are now…").
   A match returns a structured refusal **before** the LLM is called. An
   LLM-as-judge probe is documented as a stronger second layer but not
   invoked by default (latency cost on the demo path).
3. **Input — PII scrub.** Emails, AU TFNs, and credit-card numbers are
   redacted from the user question before retrieval / generation. The
   raw input is **never logged**.
4. **Retrieval — min-score refusal.** If the top fused retrieval score is
   below `RAG_MIN_SCORE=0.35`, the pipeline refuses with
   `"INSUFFICIENT_GROUNDING"`. The LLM is not called.
5. **Output — locked system prompt.** The system prompt is a module
   constant; user input is templated into the user turn only. Concatenation
   is structurally impossible.
6. **Output — citation enforcement.** The response schema is structured
   `{answer, citations[]}`. Citations are joined to the actual retrieved
   chunks by `chunk_id` before returning to the API; fabricated ids drop.
7. **Logging hygiene.** No prompts, no raw user input, no `HTTP_USER_AGENT`
   are written to logs. Refusals are logged at `WARNING` with a reason
   code only.

Every refusal returns HTTP 200 with `{"refused": true, "reason": "..."}` —
refusals are a feature, not an error.

## Alternatives considered

| Option | Why not |
|---|---|
| LLM-as-judge as the **only** injection check | Adds a second LLM call per query (≈60 s on CPU). Kept as an optional layer. |
| Open-text refusal messages | Hard to test; reason codes make refusals observable. |
| Treat refusals as 4xx errors | Conflates safety behaviour with bad input. Refusals are valid responses. |
| Microsoft Presidio for PII | Heavier dependency; the demo's regex set covers the relevant Australian PII. Presidio listed as a roadmap upgrade. |

## Consequences

**Positive**

* The golden-set test asserts `refusal_accuracy = 1.0` on three canonical
  injection prompts and `citation_grounding_score >= 0.85` on the
  knowledge questions.
* Every refusal path is unit-tested with mocks; no LLM is called when the
  system refuses.

**Negative / accepted trade-offs**

* Regex injection detection is bypassable by a determined attacker. For a
  portfolio demo this is acceptable; the production story is "add the
  LLM-as-judge layer + Azure Content Safety".

## Azure mapping

Replace the regex injection layer with **Azure AI Content Safety —
Prompt Shields**. Replace the regex PII scrub with **Presidio** behind an
Azure Function or with **Azure AI Language PII detection**. The
min-score / citation-enforcement logic stays in-process — it is application
policy, not infrastructure.
