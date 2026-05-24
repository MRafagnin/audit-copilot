# Hero: clearer narrative on the landing page

**Date**: 2026-05-25
**Status**: Completed
**Branch**: `feature/hero-narrative-rewrite`
**Files Changed**:

- `app.py` — replaced hero copy (title kept, tagline + sub-line rewritten, pills replaced).

## Motivation

The current hero reads:

> **AuditCopilot**
> Grounded answers from AUASB ASA standards and an ASX annual report, plus
> anomaly detection over journal entries with cited risk narratives.
>
> `AUASB · ASA 240 / 315 / 330`  `ASX-WOW · FY25`  `Local-first`  `Fail-closed guardrails`

Problems:

1. The tagline leads with features ("Grounded answers…") rather than audience
   and outcome. A first-time viewer cannot tell who the app is for in 5 s.
2. **"Fail-closed guardrails"** is jargon — it does not convey a concrete,
   demo-able property. The user explicitly called this out.
3. **"Local-first"** duplicates the runtime claim made elsewhere in the
   sidebar; in the hero it adds no information.
4. Nothing in the hero hints at the **hero feature** (fusion: anomaly →
   cited risk narrative), or at what the anomaly engine actually is.

## Goal

A hero that, in two short lines plus four pills, communicates:

- **Audience**: audit teams.
- **What it does**: answers questions from standards + filings; flags
  suspicious journal entries with cited explanations.
- **Why to trust it**: cites source / section / page, refuses when it
  cannot ground a response.
- **What is under the hood**: standards covered, filing in view, anomaly
  ensemble, local LLM.

## Proposed copy

- **Title** (unchanged): `AuditCopilot`
- **Tagline** (one line):
  > An AI assistant for audit teams — answers questions from auditing
  > standards and annual reports, and flags suspicious journal entries
  > with cited explanations.
- **Sub-line** (one line, smaller, same `<p>` styling):
  > Every answer cites the standard, section, and page. The system
  > refuses when it cannot ground a response.
- **Pills** (4, scannable, factual):
  1. `Standards: AUASB ASA 240 / 315 / 330 / 520 / 701`
  2. `Filing: ASX-{TICKER} · {FY}`
  3. `Anomaly engine: IsolationForest + Autoencoder + KMeans`
  4. `Runs locally · Ollama qwen2.5:7b`

The `{TICKER}` and `{FY}` interpolations reuse the existing
`_selected_company` / `_company_fy` variables already computed above the
hero block. No new state or settings.

## Rationale per change

| Change | Why |
| --- | --- |
| Lead with audience + outcome | First-time viewer sees value in one sentence, not a feature list. |
| Move the trust mechanism to its own sub-line | Citation + refusal is the differentiator vs a generic RAG demo; deserves its own line in plain English. |
| Drop "Fail-closed guardrails" pill | Jargon. Replaced by the sub-line ("refuses when it cannot ground a response"), which says the same thing in user-facing terms. |
| Drop "Local-first" pill | Already implied by the new "Runs locally · Ollama qwen2.5:7b" pill; sidebar also surfaces this. |
| Add "Anomaly engine" pill | Surfaces the second half of the product up front — currently invisible until the Scan tab. |
| Expand the ASA list | `240 / 315 / 330` undersells the corpus; the indexed standards include 520 and 701. Matches what the Ask tab examples already reference. |
| Keep pill count at 4 | Preserves current single-row layout at the default Streamlit width; no CSS work. |

## Out of scope

- No CSS changes. Existing `.ac-hero` / `.ac-pill` styles are reused.
- No new constants in `src/core/constants.py`. The standards list lives
  inline in the hero string for now; promote to a constant only if reused
  in a second place (e.g. README generation).
- No changes to tabs, sidebar, or backend.

## Implementation

Single edit in [app.py](../app.py), the hero `st.markdown(f""" … """)` block:

1. Replace the `<p>…</p>` body with the new tagline.
2. Insert a second `<p>` (or `<p class="ac-hero-sub">` if a smaller style
   is wanted — not required; the existing `<p>` styling is acceptable)
   for the trust sub-line.
3. Replace the four `<span class="ac-pill">` elements with the four
   pills listed above, keeping `html.escape()` on dynamic values
   (`_selected_company`, `_company_fy`).

No other code paths touch this string.

## Verification

1. Pre-edit search to confirm no test or doc pins the old strings:
   - `grep_search` for `Fail-closed guardrails`
   - `grep_search` for `Grounded answers from AUASB`
   - `grep_search` for `Local-first` (sidebar copy is independent — keep it there)
2. `uv run ruff check app.py` — clean.
3. `uv run mypy app.py` — clean (pure string change).
4. `uv run streamlit run app.py` against a running backend:
   - Initial load (WOW selected): hero renders, four pills on one row at
     default window width.
   - Switch ticker via sidebar dropdown to one with a different FY label
     (e.g. BHP): pill 2 updates accordingly.
   - Narrow the browser window: pills wrap cleanly (existing `.ac-pill`
     has `display: inline-block`; no overflow expected).
5. Eyeball Ask, Scan, Explain tabs — no copy referenced from them.

## Decisions resolved

1. **Tagline tone** — chose (A): "An AI assistant for audit teams — …".
   Matches live-demo framing; audience-first.
2. **Sub-line wording** — kept the harder phrasing: "refuses when it
   cannot ground a response". Matches the code's actual behaviour.
3. **Pill 3 phrasing** — kept the spelled-out form: "Anomaly engine:
   IsolationForest + Autoencoder + KMeans". Abbreviations reserved for
   the sidebar.

## Outcome

Applied to [app.py](../app.py) in the hero `st.markdown` block. `uv run
ruff check app.py` clean. No backend, CSS, or test changes.
