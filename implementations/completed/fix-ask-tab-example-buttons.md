# Bug fix: "Try one of these" example buttons do nothing

**Date**: 2026-05-23
**Status**: Completed
**Branch**: `fix/ask-tab-example-buttons`
**Files Changed**:

- `app.py` — Ask tab: rewire example-button click handler to populate the
  textarea via a pending-value indirection and auto-submit the query.

## Symptom

On the Ask tab, clicking any of the "Try one of these" example questions
showed a brief spinner (Streamlit rerun) but never produced an answer or
citations. The textarea also did not visibly update with the example text.

## Root cause

In `app.py`, inside `with tab_ask:`:

1. The textarea was declared with `key="ask_input"` and
   `value=st.session_state.get("ask_question", "")`. Once a widget is keyed,
   Streamlit owns its state under that key (`ask_input`); the `value=`
   argument is only used on the *first* render. Writes to a *different*
   key (`ask_question`) never reach the widget.
2. The example-button handler set `st.session_state["ask_question"] = example`
   then called `st.rerun()`. On the next run, the textarea read its own
   keyed state (`ask_input`), ignored `ask_question`, and rendered unchanged.
3. Even if the textarea did update, the `/ask` request was gated on
   `ask_clicked` (the primary button). Clicking an example never set
   `ask_clicked = True`, so no backend call ever fired.

Net effect: rerun spun, nothing changed, no results.

### Follow-up cause discovered during implementation

The first attempt wrote directly to `st.session_state["ask_input"]` from the
example-button handler. Because the buttons render in the right column
*after* the `text_area` (left column) is instantiated in the same script
run, Streamlit raised:

> `StreamlitAPIException: st.session_state.ask_input cannot be modified
> after the widget with key ask_input is instantiated.`

The handler must defer the write to the *next* run, before the widget is
created.

## Fix

Two-step indirection: example buttons stash the value in a separate
`ask_pending` key and rerun; a guard at the top of `tab_ask` pops
`ask_pending` into `ask_input` **before** the `text_area` widget is
instantiated. A transient `ask_autorun` flag triggers submission on the
same rerun.

### Edits in `app.py`

1. **Textarea** — dropped the `value=` argument; relies on the keyed state.

   ```python
   question = st.text_area(
       "Your question",
       placeholder="e.g. What does ASA 240 require regarding journal entry testing?",
       height=130,
       key="ask_input",
   )
   ```

2. **Pre-widget guard** — at the top of `with tab_ask:`, before any
   widgets render:

   ```python
   if "ask_pending" in st.session_state:
       st.session_state["ask_input"] = st.session_state.pop("ask_pending")
   ```

3. **Example-button loop** — write to `ask_pending` (not `ask_input`) and
   set the autorun flag:

   ```python
   for i, example in enumerate(EXAMPLE_QUESTIONS):
       if st.button(example, key=f"ex_{i}", use_container_width=True):
           st.session_state["ask_pending"] = example
           st.session_state["ask_autorun"] = True
           st.rerun()
   ```

4. **Submit gate** — treat either the primary button or the popped autorun
   flag as a trigger:

   ```python
   submit = ask_clicked or st.session_state.pop("ask_autorun", False)
   if submit and question.strip():
       ...
   ```

No other tabs were affected.

## Verification

1. `uv run streamlit run app.py`. Opened Ask tab.
2. Clicked example questions → textarea populated and `/ask` fired,
   producing answer + citations.
3. Typed custom question, clicked **Ask** → still works.
4. `uv run ruff check app.py` and `uv run ruff format --check app.py`
   clean.

## Decisions

- **Auto-submit on example click** rather than prefill-only. Better demo
  flow and matches user expectation.
- **Pending-value indirection** rather than restructuring the layout so
  buttons render before the textarea. Keeps the existing two-column layout
  intact and is the idiomatic Streamlit pattern for cross-widget writes.
- **No new tests.** The bug is in Streamlit widget wiring; the API
  contract for `/ask` is unchanged and already covered by
  `tests/api/test_main.py`.
- **Scope limited to Ask tab.**
