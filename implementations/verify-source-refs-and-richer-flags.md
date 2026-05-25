# Plan: Verify source-refs-and-richer-flags

**Date**: 2026-05-25
**Status**: Completed
**Branch**: `chore/verify-source-refs-and-richer-flags` (new — predecessor branch closed)
**Predecessor (closed)**: [completed/source-refs-and-richer-flags.md](completed/source-refs-and-richer-flags.md)

Source edits for the source-refs + richer-flags work are shipped on this branch (Phases 1–4 of the predecessor + tests). This plan owns everything that was still "Pending" there: regenerate artifacts, retrain, run the full lint/test/smoke sweep, do the Streamlit visual spot-check, and tick off the verification checklist.

**Artifacts touched (no source edits)**:

- `data/gl/journal_entries.csv` — regenerated.
- `data/flagged/top_k.csv` — regenerated.
- `data/metrics/anomaly_eval.json` — regenerated.
- `implementations/verify-source-refs-and-richer-flags.md` — checklist ticked + status flipped at the end.

## Inherited "Pending" items (from the closed predecessor)

- Phase 1 — Regenerate `data/gl/journal_entries.csv`.
- Phase 3 — Retrain via `scripts/train_anomaly.py` so `data/flagged/top_k.csv` carries the new `feature_flags` column.
- Phase 5 — `uv run pytest -q` final pass.
- Phase 5 — `uv run ruff check` + `uv run mypy src/`.
- Phase 5 — `scripts/smoke_fusion.py` eyeball.
- Phase 5 — Streamlit visual spot-check (Scan + Explain tabs).

All of the above are mapped to phases below.

## Approach

- Capture baseline ensemble metrics **before** retraining; without it, the "comparable precision@k" check is hand-wavy.
- Run lint/type/test (Phase C) in parallel with the data refresh (Phase B) — they don't depend on each other.
- Treat smoke + Streamlit (Phase D) as the LLM-touching verifications; if Ollama isn't running, surface that early rather than skip silently.
- One commit at the end, on the existing branch. No push without explicit approval.

## Resolved decisions

- **Precision@k acceptance band**: ±0.05 absolute on the ensemble. The `desc_len` feature distribution narrowed (all rows ~30-char references now), so small drift is expected; a >0.05 drop warrants a diagnostic, not an automatic accept.
- **New branch.** Work on `chore/verify-source-refs-and-richer-flags` (predecessor `feature/source-refs-and-richer-flags` was closed). One Conventional Commit closes the sweep.
- **Out of scope**: source edits, new dependencies, Chroma reindex (corpus unchanged).
- **Streamlit visual is human-driven by default.** Playwright-driven automation is available on request but slower for what amounts to visual judgments.

## Steps

### Phase A — Baseline + data refresh

1. Snapshot pre-change ensemble metrics from `data/metrics/anomaly_eval.json` (in-memory only — `precision_at_k`, `roc_auc`, `pr_auc`).
2. Regenerate GL: `uv run python scripts/gen_journal_entries.py`. *(closes predecessor Phase 1 / step 5)*
3. Spot-check `data/gl/journal_entries.csv` — every `description` matches `^Note \d+ · p\.\d+ — .+$`; sample rows align with `ACCOUNT_NOTE_MAP` (e.g., `4000-Revenue` → Note 6, pages 120–144).

### Phase B — Retrain + flagged-CSV check (*depends on A*)

4. Retrain: `uv run python scripts/train_anomaly.py`. Watch the calibrated weights and per-detector metrics in the logs. *(closes predecessor Phase 3 / step 14)*
5. Compare new `data/metrics/anomaly_eval.json` ensemble values vs the step-1 snapshot. Accept within ±0.05 absolute of baseline `precision_at_k`; no detector collapses to ~0.
6. Inspect `data/flagged/top_k.csv`:
   - `feature_flags` column populated for ≥90% of rows.
   - Rows with `anomaly_type == "unusual_user_account"` carry `is_unusual_user_account`.
   - Rows with `anomaly_type == "near_duplicate"` carry `is_near_duplicate`.
   - ≥4 distinct flag tokens across the top 20 rows.

### Phase C — Lint + type + tests (*parallel with B; runnable any time after source edits*)

7. `uv run ruff check` — clean. *(closes predecessor Phase 5 lint slice)*
8. `uv run mypy src/` — clean. *(closes predecessor Phase 5 type slice)*
9. `uv run pytest -q --cov=src --cov-report=term-missing` — passes; ≥80% on changed files (`src/fusion/explain.py`, `scripts/train_anomaly.py` via `tests/anomaly/test_train_enrichment.py`). *(closes predecessor Phase 5 / step 20)*

### Phase D — Smoke + visual sweep (*depends on B*)

10. `uv run python scripts/smoke_fusion.py` (needs Ollama on `localhost:11434` with `qwen2.5:7b-instruct` + indexed `data/chroma/`). Confirm: *(closes predecessor step 21)*
    - Narrative cites ≥1 `[n]` tag.
    - Prompt block (printed or briefly instrumented) shows `source_ref: "..."`.
    - Narrative language reflects active flags (e.g., "unusual user/account", "near-duplicate", "round-dollar to revenue").
11. Streamlit (`make run` or `scripts/run_dev.ps1`), open `http://localhost:8501`: *(closes predecessor Phase 5 Streamlit slice)*
    - **Scan tab**: column header "Source ref"; KPI strip = `Flagged shown` / `Mean ensemble score` / `Large amount %` / `Unusual user/account %`; ≥4 distinct flag badges across the top 20 rows; severity-first order (`Near duplicate` / `Amount outlier` leftmost).
    - **Explain tab**: pick a row tagged `is_unusual_user_account` or `is_near_duplicate`, generate the narrative; LLM cites standards and references the relevant control concern.

### Phase E — Close out

12. Tick all boxes in the Verification checklist below.
13. Single Conventional Commit on the new branch:
    ```
    chore(verify): regen GL + retrain + lint/test sweep for source-refs-and-richer-flags
    ```
    Do not push without explicit user approval.

## Verification checklist

(Inherits the predecessor's checklist verbatim plus baseline + commit items.)

- [x] Pre-retrain ensemble metrics snapshotted.
- [x] `uv run python scripts/gen_journal_entries.py` succeeds; regenerated GL CSV: every `description` is a `Note N · p.NNN — Title` reference.
- [x] `uv run python scripts/train_anomaly.py` succeeds; calibrated weights logged; ensemble `precision_at_k` within ±0.05 of baseline.
- [x] Regenerated flagged CSV: `feature_flags` column populated; values match `anomaly_type` for `unusual_user_account` and `near_duplicate` rows.
- [x] `uv run ruff check` clean.
- [x] `uv run mypy src/` clean.
- [x] `uv run pytest -q --cov=src` passes; coverage ≥80% on changed files.
- [x] `scripts/smoke_fusion.py` narrative cites `[n]` tags and references active flags; `source_ref:` present in the prompt block.
- [x] Streamlit Scan tab: column header is "Source ref"; ≥4 distinct flag badges visible across top 20 rows; KPI strip shows the new metrics; severity-first order.
- [x] Streamlit Explain tab: narrative for an `is_unusual_user_account` or `is_near_duplicate` row references the relevant control concern; LLM treats `source_ref` as a citation.
- [x] `data/metrics/anomaly_eval.json` precision@k comparable to pre-change values (no major regression).
- [x] Verification commit landed on `chore/verify-source-refs-and-richer-flags`; not pushed.

## Further considerations

1. **If Ollama isn't running** — start it via the documented Windows service command before Phase D and proceed; if start fails, stop and ask the user.
2. **If precision@k regresses by >0.05** — diagnose first (likely candidate: `desc_len` is now near-constant and contributes noise; consider dropping it). Don't accept silently; don't revert without a quick diagnostic.
3. **Streamlit visual** — human-driven by default. Playwright automation available on request.

## Excluded

- No source edits in `src/`, `scripts/`, `app.py`, or tests.
- No new dependencies.
- No rebuild of `data/chroma/` (corpus unchanged).
- No push to remote.
