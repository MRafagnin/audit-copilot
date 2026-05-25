# Plan: Annual-report source refs + richer risk flags

**Date**: 2026-05-25
**Status**: Closed — source edits shipped. Verification + remaining checklist items handed off to [verify-source-refs-and-richer-flags.md](verify-source-refs-and-richer-flags.md).
**Branch**: `feature/source-refs-and-richer-flags` (off `main`)

> **Handoff (2026-05-25)**: Phases 1–4 source edits and Phase 5 tests are done. All remaining "Pending" rows in the Status snapshot and every unchecked item in the Verification checklist below are tracked in the successor plan and will be ticked off there. This file is kept for design/decision history only — do not edit further.

**Files to change**:

- `scripts/gen_journal_entries.py` — `ACCOUNT_NOTE_MAP`, `_source_ref` helper; swap free-text descriptions for annual-report references.
- `scripts/train_anomaly.py` — compute and persist a `feature_flags` column on the top-k flagged CSV (includes cross-row signals).
- `src/fusion/explain.py` — extend `derive_feature_flags` (more single-row flags); prefer persisted `feature_flags` column in `flagged_transaction_from_row`; rename prompt label `description` → `source_ref`; extend `_FLAG_QUERY_TERMS`.
- `app.py` — extend `FLAG_LABELS` with audit-toned colours for new flags; relabel Scan table column "Description" → "Source ref"; adjust `_scan_kpis` to surface the new signals.
- `tests/fusion/test_explain.py` — cover new flags, persisted-column path, and prompt label.
- `tests/anomaly/` *(new test or existing file)* — unit-test the cross-row flag enrichment in the train script.
- `data/gl/journal_entries.csv`, `data/flagged/top_k.csv` — regenerated artifacts.

```powershell
git checkout -b feature/source-refs-and-richer-flags
```

Two changes shipped together because they share the same regeneration cycle (regen GL → retrain → refresh flagged CSV) and both target the Scan UI / fusion-explainer prompt quality.

1. **Source refs** — replace Lorem-ipsum-style faker descriptions ("Threat article writer break form side.") with account-aware annual-report references like `"Note 11 · p.178 — Trade and other payables"`. Format is a citation, not prose.
2. **Richer flags** — the Scan UI currently shows only three single-row flags (`is_weekend`, `is_after_hours`, `is_round_amount`) because cross-row context isn't carried through to the API. Extend the single-row flag set and persist cross-row flags into `data/flagged/top_k.csv` at training time so the UI and LLM both see the full risk picture.

## Approach

### Source refs

- Keep the column name `description` everywhere (`FlaggedTxOut.description`, `features.py:desc_len`, fusion prompt, tests). Content semantics change; structure doesn't. Avoids a cross-cutting schema rename.
- Module-level `ACCOUNT_NOTE_MAP: dict[str, tuple[int, str, range]]` in `scripts/gen_journal_entries.py` mapping each of the 12 accounts to `(note_number, note_title, plausible_page_range)`.
- Page numbers are seeded by the existing `rng` — same `--seed` reproduces the same CSV.
- No ticker baked into the CSV. GL stays company-agnostic so the sidebar ticker switcher doesn't produce stale references.
- Page numbers are plausible, not real. Documented in the script docstring as synthetic — we are not pretending to resolve them against the indexed PDF.

### Richer flags

- **Single-row extensions** in `derive_feature_flags` (computable from one row + account):
  - `is_benford_first_digit_9` — leading digit of `|debit - credit|` equals 9. Matches the seeded Benford-violation pattern.
  - `is_large_amount` — `|amount| >= 100_000`.
  - `is_sensitive_account` — account in `{4000-Revenue, 3000-Equity, 5000-COGS, 2000-AccountsPayable}`.
  - `is_round_credit_to_revenue` — round amount AND credit > 0 AND account = `4000-Revenue`. Classic posting-fraud signature.
- **Cross-row flags** computed in `scripts/train_anomaly.py` from the feature matrix and persisted into `data/flagged/top_k.csv` as a `feature_flags` column (semicolon-joined):
  - `is_unusual_user_account` — `user_account_freq` in the bottom 5% across the full corpus.
  - `is_amount_outlier_for_account` — `|amount_zscore_per_account| > 3`.
  - `is_near_duplicate` — same (account, amount_abs) with another row within ±15 min posting_ts.
- `flagged_transaction_from_row` prefers the persisted `feature_flags` column when present (parses the semicolon list). Falls back to `derive_feature_flags` for callers that hand-build rows (tests, smoke scripts).
- Training script is the single source of truth for cross-row flags; the API never recomputes them at request time.

## Resolved decisions

- **Column name `description` stays.** Renaming to `source_ref` would touch the API schema, every test fixture, and the feature builder for cosmetic gain. UI relabels the column header instead.
- **Prompt label changes** to `source_ref: "..."` in `format_transaction_block` so the LLM treats the field as evidence reference. Dataclass field name stays `description` — purely a string-literal change in the prompt template.
- **No ticker in the source ref**, no real page lookups. Synthetic demo data; honest scope.
- **Cross-row flags persisted at training time**, not recomputed in the API. Avoids re-implementing groupby logic at request time and keeps the API hot path simple.
- **Thresholds** ($100k, bottom 5%, |z| > 3, ±15 min) live as module-level constants in their respective files — audit-domain numbers, not user-tunable settings.
- **Flag render order is severity-first** in the UI (`is_near_duplicate`, `is_amount_outlier_for_account`, `is_unusual_user_account`, `is_round_credit_to_revenue` before the milder weekend/after-hours/round signals) so the most damning badge sits leftmost.
- **Retrain** rather than reuse old models — `desc_len` distribution narrows substantially (all rows now ~30-char references). Recalibrating is cleaner than mixing old artifacts with new data.

## Status snapshot (2026-05-25)

| Phase | Item | Status |
| --- | --- | --- |
| 1 | `ACCOUNT_NOTE_MAP` + `_source_ref` in `gen_journal_entries.py` | Done |
| 1 | `_build_row` uses `_source_ref`; module docstring updated | Done |
| 1 | Regenerate `data/gl/journal_entries.csv` | Pending |
| 2 | `derive_feature_flags` accepts `account=`; new single-row flags | Done |
| 2 | `_LARGE_AMOUNT_THRESHOLD`, `_SENSITIVE_ACCOUNTS` constants | Done |
| 2 | `_FLAG_QUERY_TERMS` extended | Done |
| 2 | `flagged_transaction_from_row` prefers persisted `feature_flags` | Done |
| 2 | `format_transaction_block` emits `source_ref:` | Done |
| 3 | `enrich_feature_flags` + thresholds in `train_anomaly.py` | Done |
| 3 | Severity-ordered `_FLAG_ORDER`; persisted to `top_k.csv` | Done |
| 3 | Retrain (`scripts/train_anomaly.py`) | Pending |
| 4 | `FLAG_LABELS` extended with severity-first ordering | Done |
| 4 | Scan table column relabeled "Source ref" | Done |
| 4 | `_scan_kpis` swapped to new metric mix | Done |
| 5 | `tests/fusion/test_explain.py` — new flags, persisted column, `source_ref` label | Done |
| 5 | `tests/anomaly/test_train_enrichment.py` — cross-row enrichment | Done |
| 5 | `uv run pytest -q` final pass | Pending |
| 5 | `uv run ruff check` + `uv run mypy src/` | Pending |
| 5 | `scripts/smoke_fusion.py` eyeball | Pending |
| 5 | Streamlit visual spot-check (Scan + Explain tabs) | Pending |

Remaining work is verification only — regenerate data, retrain, and run the lint/test/smoke sweep. No further source edits planned in this scope.

## Steps

### Phase 1 — Source refs

1. In `scripts/gen_journal_entries.py`, add `ACCOUNT_NOTE_MAP` near the top:

   ```python
   ACCOUNT_NOTE_MAP: dict[str, tuple[int, str, range]] = {
       "1000-Cash":                 (8,  "Cash and cash equivalents",                range(150, 165)),
       "1100-AccountsReceivable":   (9,  "Trade and other receivables",              range(160, 175)),
       "1200-Inventory":            (10, "Inventories",                              range(165, 180)),
       "2000-AccountsPayable":      (11, "Trade and other payables",                 range(170, 185)),
       "2100-AccruedLiabilities":   (12, "Provisions and accrued liabilities",       range(175, 190)),
       "3000-Equity":               (20, "Contributed equity and reserves",          range(210, 225)),
       "4000-Revenue":              (6,  "Revenue from contracts with customers",   range(120, 145)),
       "5000-COGS":                 (7,  "Cost of goods sold",                       range(140, 155)),
       "6000-OperatingExpense":     (13, "Operating expenses",                       range(180, 195)),
       "6100-Travel":               (14, "Employee and travel expenses",             range(185, 200)),
       "6200-Marketing":            (15, "Marketing and selling expenses",           range(190, 205)),
       "7000-InterestExpense":      (18, "Finance costs",                            range(200, 215)),
   }
   ```

2. Add helper:

   ```python
   def _source_ref(rng: random.Random, account: str) -> str:
       note, title, pages = ACCOUNT_NOTE_MAP[account]
       page = rng.choice(list(pages))
       return f"Note {note} · p.{page} — {title}"
   ```

3. In `_build_row`, replace `"description": faker.sentence(nb_words=6)` with `"description": _source_ref(rng, account)`. Remove the `faker` import / usage if no longer needed (verify before deleting).
4. Update the module docstring — note that `description` is a synthetic annual-report reference, not free-text narration.
5. Regenerate: `uv run python scripts/gen_journal_entries.py`.

### Phase 2 — Single-row flag extensions

6. In `src/fusion/explain.py`, extend the signature:

   ```python
   def derive_feature_flags(
       *, posting_ts: str, debit: float, credit: float, account: str
   ) -> tuple[str, ...]:
   ```

7. Add module-level constants:

   ```python
   _LARGE_AMOUNT_THRESHOLD: float = 100_000.0
   _SENSITIVE_ACCOUNTS: frozenset[str] = frozenset({
       "4000-Revenue", "3000-Equity", "5000-COGS", "2000-AccountsPayable",
   })
   ```

8. Inside the function, after the existing 3 flags, add:
   - `is_benford_first_digit_9` — reuse `_first_digit` from `src/anomaly/features.py` (or inline a small helper).
   - `is_large_amount` — `amount >= _LARGE_AMOUNT_THRESHOLD`.
   - `is_sensitive_account` — `account in _SENSITIVE_ACCOUNTS`.
   - `is_round_credit_to_revenue` — `is_round_amount AND credit > 0 AND account == "4000-Revenue"`.
9. Extend `_FLAG_QUERY_TERMS` with retrieval phrases for the new flags (audit-standard language: e.g., `is_large_amount → "high-value journal entry materiality threshold"`, `is_sensitive_account → "fraud risk revenue equity manual journal entries"`, `is_round_credit_to_revenue → "round-dollar revenue credit fictitious sales"`, `is_amount_outlier_for_account → "outlier amount account population analytical procedure"`).
10. Update `flagged_transaction_from_row`:

    ```python
    raw_flags = row.get("feature_flags")
    if isinstance(raw_flags, str) and raw_flags.strip():
        flags = tuple(f for f in raw_flags.split(";") if f)
    else:
        flags = derive_feature_flags(
            posting_ts=posting_ts, debit=debit, credit=credit, account=account,
        )
    ```

11. Rename the prompt label in `format_transaction_block`:

    ```python
    f'  source_ref: "{safe_description}"\n'   # was: description:
    ```

### Phase 3 — Cross-row flag enrichment in training

12. In `scripts/train_anomaly.py`, after the top-k slice is built, compute the full flag set for those rows:
    - Reuse the already-computed `fm.X` columns by index (positions of `is_round_amount`, `benford_first_digit`, `is_after_hours`, `is_weekend`, `user_account_freq`, `amount_zscore_per_account` are stable per `FEATURE_COLUMNS`).
    - Determine the bottom-5% threshold for `user_account_freq` from the **full** matrix `fm.X`, not just top-k.
    - Detect near-duplicates with a pandas groupby on `(account, amount_abs_rounded)` over the full GL frame; mark rows whose nearest neighbour in the group is within 15 minutes.
    - Build a per-row `feature_flags` string (semicolon-joined, severity-ordered) and assign it to `flagged["feature_flags"]` before `to_csv`.
13. Add module-level thresholds at the top of `train_anomaly.py`:

    ```python
    _UNUSUAL_PAIR_QUANTILE: float = 0.05
    _ZSCORE_OUTLIER: float = 3.0
    _NEAR_DUP_WINDOW_MIN: int = 15
    ```

14. Retrain: `uv run python scripts/train_anomaly.py`.

### Phase 4 — UI

15. In `app.py`, extend `FLAG_LABELS` (severity-first ordering reflected in the dict insertion order — Python 3.7+ preserves it):

    ```python
    FLAG_LABELS = {
        "is_near_duplicate":            ("Near duplicate",       "#b91c1c"),
        "is_amount_outlier_for_account":("Amount outlier",       "#b91c1c"),
        "is_unusual_user_account":      ("Unusual user/account", "#b45309"),
        "is_round_credit_to_revenue":   ("Round credit→revenue", "#b45309"),
        "is_benford_first_digit_9":     ("Benford 9",            "#a16207"),
        "is_large_amount":              ("Large amount",         "#0b2545"),
        "is_sensitive_account":         ("Sensitive account",    "#13315c"),
        "is_weekend":                   ("Weekend",              "#7c3aed"),
        "is_after_hours":               ("After hours",          "#0891b2"),
        "is_round_amount":              ("Round amount",         "#d97706"),
    }
    ```

16. In the Scan table `column_config`, relabel:

    ```python
    "description": st.column_config.TextColumn("Source ref", width="medium"),
    ```

17. Update `_scan_kpis` — keep 4 KPIs, swap to the more demo-worthy mix:
    - Flagged shown (count)
    - Mean ensemble score
    - % large amount (`is_large_amount`)
    - % unusual user/account (`is_unusual_user_account`)

### Phase 5 — Tests + verification

18. Update `tests/fusion/test_explain.py`:
    - New parametrize cases for each new single-row flag in `derive_feature_flags`.
    - Update existing call sites to pass `account=...`.
    - Add a case for `flagged_transaction_from_row` reading a `feature_flags` column.
    - Add a case asserting the prompt block contains `source_ref:` (not `description:`).
19. Add a unit test for the train-script enrichment: build a tiny synthetic frame, call the enrichment helper, assert `feature_flags` column populated with expected values.
20. Run `uv run pytest -q`. Coverage on changed files must stay ≥80%.
21. Smoke: `uv run python scripts/smoke_fusion.py`. Confirm the narrative references the new flags (e.g., "unusual user-account pairing") and treats `source_ref` as a citation.
22. Eyeball regenerated artifacts:
    - First ~30 rows of `data/gl/journal_entries.csv` — descriptions look like `Note N · p.NNN — Title`.
    - `data/flagged/top_k.csv` — `feature_flags` column populated; sanity-check that rows with `anomaly_type=unusual_user_account` carry `is_unusual_user_account`.

## Verification checklist

- [ ] `uv run pytest -q` passes; coverage ≥80% on changed files.
- [ ] `uv run ruff check` and `uv run mypy src/` clean.
- [ ] Regenerated GL CSV: every `description` is a `Note N · p.NNN — Title` reference.
- [ ] Regenerated flagged CSV: `feature_flags` column populated; values match `anomaly_type` for ground-truth rows in the spot-check.
- [ ] Streamlit Scan tab: column header is "Source ref"; ≥4 distinct flag badges visible across top 20 rows; KPI strip shows the new metrics.
- [ ] Streamlit Explain tab: narrative for a `is_unusual_user_account` or `is_near_duplicate` row references the relevant control concern; LLM treats `source_ref` as a citation.
- [ ] `data/metrics/anomaly_eval.json` precision@k comparable to pre-change values (no major regression).

## Excluded

- No API schema rename (`description` field name stays).
- No real page lookups against the indexed AR PDF (deferred — would couple GL generation to per-company indexing).
- No new ML features or model architecture changes — only retrain on the regenerated data.
- No new endpoints; existing `feature_flags` field carries the richer set.
- No new dependencies.
