"""Streamlit demo UI for AuditCopilot.

Three tabs (`Ask`, `Scan`, `Explain`) backed by the FastAPI service. The UI
is styled for an audit / finance audience — restrained palette, citation
cards, KPI strip, feature-flag badges.

Run::

    uv run streamlit run app.py

The backend URL defaults to ``http://localhost:8000`` and can be overridden
through the ``AUDIT_COPILOT_API`` environment variable.
"""

from __future__ import annotations

import html
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("AUDIT_COPILOT_API", "http://localhost:8000")
REQUEST_TIMEOUT = httpx.Timeout(180.0, connect=5.0)
INGEST_TIMEOUT = httpx.Timeout(600.0, connect=5.0)

EXAMPLE_QUESTION_TEMPLATES = (
    "What does ASA 240 require regarding journal entry testing?",
    "How should auditors respond to identified risks of material misstatement?",
    "What does ASA 315 say about understanding the entity and its environment?",
    "What are the auditor's responsibilities for fraud risk assessment under ASA 240?",
    "How does ASA 330 require auditors to design responses to assessed risks?",
    "What does ASA 520 require for analytical procedures during the audit?",
    "When is a matter considered a key audit matter under ASA 701?",
    "What does the {name} annual report disclose about segment performance?",
    "What significant accounting estimates does {name} disclose in its latest annual report?",
    "What related-party transactions are disclosed in the {name} annual report?",
)

FLAG_LABELS = {
    "is_weekend": ("Weekend", "#7c3aed"),
    "is_after_hours": ("After hours", "#0891b2"),
    "is_round_amount": ("Round amount", "#d97706"),
}

st.set_page_config(
    page_title="AuditCopilot",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
<style>
:root {
  --ac-navy: #0b2545;
  --ac-navy-2: #13315c;
  --ac-teal: #1d7874;
  --ac-mist: #e8edf3;
  --ac-ink: #1f2937;
  --ac-muted: #6b7280;
  --ac-danger: #b91c1c;
  --ac-warn: #b45309;
  --ac-ok: #047857;
}

/* Tighten the top padding so the hero sits up */
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; }

/* Hero header */
.ac-hero {
  background: linear-gradient(135deg, var(--ac-navy) 0%, var(--ac-navy-2) 60%, var(--ac-teal) 100%);
  color: white;
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 10px 30px -12px rgba(11, 37, 69, 0.45);
}
.ac-hero h1 { color: white; margin: 0; font-size: 1.7rem; letter-spacing: 0.2px; }
.ac-hero p  { color: #cfe0f1; margin: 0.25rem 0 0; font-size: 0.92rem; }

/* Pills */
.ac-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.2px;
  margin-right: 6px;
  background: rgba(255,255,255,0.14);
  color: white;
  border: 1px solid rgba(255,255,255,0.25);
}

/* Status dot */
.ac-status { display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; }
.ac-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.ac-dot-ok   { background: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,0.18); }
.ac-dot-bad  { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.18); }

/* Citation card */
.ac-cite {
  border: 1px solid #e5e7eb;
  background: #fbfcfd;
  border-left: 4px solid var(--ac-teal);
  border-radius: 10px;
  padding: 0.6rem 0.8rem;
  margin-bottom: 0.5rem;
}
.ac-cite .tag {
  display: inline-block;
  min-width: 22px;
  text-align: center;
  background: var(--ac-navy);
  color: white;
  font-weight: 700;
  font-size: 0.72rem;
  border-radius: 6px;
  padding: 1px 6px;
  margin-right: 8px;
}
.ac-cite .src    { font-weight: 600; color: var(--ac-ink); }
.ac-cite .sec    { color: var(--ac-muted); font-size: 0.85rem; margin-left: 6px; }
.ac-cite .chunk  { color: #94a3b8; font-size: 0.74rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 2px; }

/* Narrative card */
.ac-narrative {
  background: #fbfcfd;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  line-height: 1.55;
  color: var(--ac-ink);
}

/* Refusal / info banners */
.ac-refusal {
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-left: 4px solid var(--ac-warn);
  border-radius: 10px;
  padding: 0.6rem 0.8rem;
  color: #7c2d12;
  font-size: 0.92rem;
}
.ac-refusal b { color: var(--ac-warn); }

/* Feature-flag badge */
.ac-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  margin-right: 4px;
  color: white;
}

/* Tabs */
.stTabs [role="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--ac-navy) !important; }

/* Sidebar — leave background to Streamlit's theme so dark/light both work */
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _post(
    path: str, payload: dict[str, Any], *, timeout: httpx.Timeout | None = None
) -> dict[str, Any] | None:
    """POST to the backend and surface errors as Streamlit alerts."""
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout or REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
        return None
    if response.status_code >= 400:
        st.error(f"{path} returned {response.status_code}: {response.text}")
        return None
    return response.json()


def _get(path: str) -> dict[str, Any] | None:
    """GET from the backend and surface errors as Streamlit alerts."""
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
        return None
    if response.status_code >= 400:
        st.error(f"{path} returned {response.status_code}: {response.text}")
        return None
    return response.json()


def _fetch_companies() -> list[dict[str, Any]]:
    """Pull the company allowlist from the backend (empty on failure)."""
    data = _get("/companies")
    if data is None:
        return []
    items = data.get("items", [])
    return list(items) if isinstance(items, list) else []


def _check_health() -> tuple[bool, str]:
    """Probe ``/health`` and return ``(ok, message)``.

    Uses a short connect timeout so the sidebar stays responsive when the
    backend is down.
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=httpx.Timeout(3.0, connect=1.5))
    except httpx.HTTPError as exc:
        return False, f"unreachable ({exc.__class__.__name__})"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}"
    return True, "online"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_citations(citations: list[dict[str, Any]]) -> None:
    """Render a citation list as styled cards."""
    if not citations:
        st.caption("No citations.")
        return
    for c in citations:
        page = c.get("page")
        page_html = (
            f'<span class="sec">· p.{html.escape(str(page))}</span>' if page is not None else ""
        )
        st.markdown(
            f"""
            <div class="ac-cite">
              <div>
                <span class="tag">{html.escape(str(c["tag"]))}</span>
                <span class="src">{html.escape(str(c["source"]))}</span>
                <span class="sec">— {html.escape(str(c["section"]))}</span>
                {page_html}
              </div>
              <div class="chunk">{html.escape(str(c["chunk_id"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_refusal(reason: str) -> None:
    """Render a refusal banner with the supplied reason code."""
    st.markdown(
        f'<div class="ac-refusal"><b>Refused</b> — {html.escape(reason)}</div>',
        unsafe_allow_html=True,
    )


def _render_flags(flags: list[str]) -> str:
    """Render feature flags as inline coloured badges (returns HTML)."""
    if not flags:
        return '<span style="color:#9ca3af;font-size:0.8rem;">—</span>'
    parts = []
    for flag in flags:
        label, color = FLAG_LABELS.get(flag, (flag, "#475569"))
        parts.append(
            f'<span class="ac-badge" style="background:{color};">{html.escape(label)}</span>'
        )
    return "".join(parts)


def _scan_df(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Transform raw scan items into a display-ready DataFrame."""
    df = pd.DataFrame(items)
    df["amount"] = df["debit"].where(df["debit"] > 0, df["credit"])
    df["flags"] = df["feature_flags"].apply(lambda fs: ", ".join(fs) if fs else "")
    return df[
        [
            "tx_id",
            "date",
            "account",
            "user",
            "amount",
            "ensemble_score",
            "flags",
            "description",
        ]
    ]


def _scan_kpis(items: list[dict[str, Any]]) -> None:
    """Render a 4-column KPI strip for the current scan result."""
    if not items:
        return
    df = pd.DataFrame(items)
    weekend_pct = 100.0 * df["feature_flags"].apply(lambda fs: "is_weekend" in fs).mean()
    after_hours_pct = 100.0 * df["feature_flags"].apply(lambda fs: "is_after_hours" in fs).mean()
    round_pct = 100.0 * df["feature_flags"].apply(lambda fs: "is_round_amount" in fs).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flagged shown", f"{len(df):,}")
    c2.metric("Mean ensemble score", f"{df['ensemble_score'].mean():.3f}")
    c3.metric("Weekend %", f"{weekend_pct:.0f}%")
    c4.metric("After-hours / Round %", f"{after_hours_pct:.0f}% · {round_pct:.0f}%")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## AuditCopilot")
    st.caption("Local-first AI for Audit & Assurance")

    ok, msg = _check_health()
    dot = "ac-dot-ok" if ok else "ac-dot-bad"
    st.markdown(
        f'<div class="ac-status"><span class="ac-dot {dot}"></span>'
        f"<span><b>Backend:</b> {html.escape(msg)}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"`{API_URL}`")

    st.divider()
    st.markdown("### Company")
    companies = _fetch_companies() if ok else []
    company_lookup = {c["ticker"]: c for c in companies}
    tickers = list(company_lookup.keys())
    if tickers:
        default_idx = tickers.index("WOW") if "WOW" in tickers else 0
        prior = st.session_state.get("company")
        if prior in tickers:
            default_idx = tickers.index(prior)

        def _format_ticker(t: str) -> str:
            c = company_lookup[t]
            marker = "" if c["indexed"] else " · (not indexed)"
            return f"{t} — {c['name']} ({c['fy_label']}){marker}"

        selected = st.selectbox(
            "ASX ticker",
            tickers,
            index=default_idx,
            format_func=_format_ticker,
            key="company_select",
        )
        st.session_state["company"] = selected
        selected_info = company_lookup[selected]
        if not selected_info["indexed"]:
            st.warning(f"{selected} not yet indexed.")
            if st.button(
                f"Ingest {selected}", type="primary", use_container_width=True, key="ingest_btn"
            ):
                with st.spinner(f"Fetching and indexing {selected} annual report…"):
                    result = _post(f"/companies/{selected}/ingest", {}, timeout=INGEST_TIMEOUT)
                if result is not None:
                    st.success(
                        f"Indexed {result['chunks_added']} chunks ({result['took_ms']} ms)."
                        if not result.get("cached")
                        else "Already indexed."
                    )
                    st.rerun()
        else:
            st.caption(f"{selected_info['name']} · {selected_info['fy_label']} · indexed")
    else:
        st.info("Company list unavailable.")
        st.session_state.setdefault("company", "WOW")

    st.divider()
    st.markdown("### Stack")
    st.markdown(
        "- **RAG**: BM25 + dense (RRF)\n"
        "- **LLM**: Ollama `qwen2.5:7b-instruct`\n"
        "- **Anomaly**: IsolationForest + AE + KMeans\n"
        "- **Index**: ChromaDB (embedded)"
    )

    st.divider()
    st.markdown("### Safety")
    st.markdown(
        "- Length cap · injection regex\n"
        "- PII scrub\n"
        "- Refuse on low retrieval\n"
        "- Locked system prompt\n"
        "- Citation enforcement"
    )

    st.divider()
    st.caption("See `docs/adr/` for design decisions.")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

_selected_company: str = st.session_state.get("company", "WOW")
_company_info: dict[str, Any] = company_lookup.get(_selected_company, {})  # type: ignore[has-type]
_company_name: str = str(_company_info.get("name", _selected_company))
_company_fy: str = str(_company_info.get("fy_label", ""))

st.markdown(
    f"""
    <div class="ac-hero">
      <h1>AuditCopilot</h1>
      <p>
        Grounded answers from AUASB ASA standards and an ASX annual report,
        plus anomaly detection over journal entries with cited risk narratives.
      </p>
      <div style="margin-top:0.55rem;">
        <span class="ac-pill">AUASB · ASA 240 / 315 / 330</span>
        <span class="ac-pill">ASX-{html.escape(_selected_company)} · {html.escape(_company_fy)}</span>
        <span class="ac-pill">Local-first</span>
        <span class="ac-pill">Fail-closed guardrails</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = tuple(q.format(name=_company_name) for q in EXAMPLE_QUESTION_TEMPLATES)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ask, tab_scan, tab_explain = st.tabs(
    ["💬  Ask", "🔎  Scan journal entries", "🧭  Explain anomaly"]
)

# --- Ask ---------------------------------------------------------------------

with tab_ask:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Ask a grounded question")
        st.caption(
            f"Hybrid retrieval over AUASB ASA + ASX-{_selected_company}. Every answer is cited; "
            "the system refuses when retrieval confidence is below threshold."
        )
        question = st.text_area(
            "Your question",
            value=st.session_state.get("ask_question", ""),
            placeholder="e.g. What does ASA 240 require regarding journal entry testing?",
            height=130,
            key="ask_input",
        )
        ask_clicked = st.button(
            "Ask", type="primary", disabled=not question.strip(), use_container_width=True
        )

    with right:
        st.markdown("**Try one of these**")
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state["ask_question"] = example
                st.rerun()

    if ask_clicked:
        with st.spinner("Retrieving standards and generating a grounded answer…"):
            data = _post(
                "/ask",
                {"question": question.strip(), "company": _selected_company},
            )
        if data is not None:
            if data["refused"]:
                _render_refusal(str(data["reason"]))
            ans_col, cite_col = st.columns([3, 2], gap="large")
            with ans_col:
                st.markdown("#### Answer")
                st.markdown(
                    f'<div class="ac-narrative">{html.escape(str(data["answer"])).replace(chr(10), "<br/>")}</div>',
                    unsafe_allow_html=True,
                )
            with cite_col:
                st.markdown("#### Citations")
                _render_citations(data["citations"])

# --- Scan --------------------------------------------------------------------

with tab_scan:
    st.subheader("Top flagged journal entries")
    st.caption(
        "Ensemble of IsolationForest + autoencoder + KMeans over 50 000 synthetic GL rows "
        "(seed 42). Score is min-max normalised; weights calibrated against PR-AUC."
    )

    controls = st.columns([1, 1, 6])
    with controls[0]:
        n = st.number_input("Rows", min_value=5, max_value=200, value=10, step=5)
    with controls[1]:
        scan_clicked = st.button("Scan", type="primary", use_container_width=True)

    if scan_clicked:
        with st.spinner("Loading flagged transactions…"):
            data = _post("/scan", {"n": int(n)})
        if data is not None:
            items = data["items"]
            st.session_state["last_scan_items"] = items
            st.session_state["last_scan_ids"] = [item["tx_id"] for item in items]

    items = st.session_state.get("last_scan_items", [])
    if not items:
        st.info("Run a scan to load flagged transactions.")
    else:
        _scan_kpis(items)
        st.markdown("##### Flagged transactions")
        df = _scan_df(items)
        max_score = float(df["ensemble_score"].max() or 1.0)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "tx_id": st.column_config.TextColumn("Tx id", width="small"),
                "date": st.column_config.TextColumn("Date", width="small"),
                "account": st.column_config.TextColumn("Account", width="small"),
                "user": st.column_config.TextColumn("User", width="small"),
                "amount": st.column_config.NumberColumn(
                    "Amount", format="$%.2f", help="Debit or credit amount"
                ),
                "ensemble_score": st.column_config.ProgressColumn(
                    "Score",
                    help="Normalised ensemble anomaly score",
                    format="%.3f",
                    min_value=0.0,
                    max_value=max_score,
                ),
                "flags": st.column_config.TextColumn("Flags", width="medium"),
                "description": st.column_config.TextColumn("Description", width="large"),
            },
        )
        st.caption(
            "Pick any `tx_id` from this list in the **Explain** tab to generate a narrative."
        )

# --- Explain -----------------------------------------------------------------

with tab_explain:
    st.subheader("Grounded narrative for a flagged transaction")
    st.caption(
        "The explainer retrieves audit-standard chunks relevant to the transaction's feature "
        "flags and asks the LLM to produce a plain-English risk narrative with citations. "
        "Refuses rather than hallucinating when grounding is weak."
    )

    options = st.session_state.get("last_scan_ids", [])
    cols = st.columns([3, 1])
    with cols[0]:
        if options:
            tx_id = st.selectbox("Transaction id (from latest scan)", options, key="explain_select")
        else:
            tx_id = st.text_input(
                "Transaction id",
                placeholder="Run a scan first or paste a tx_id",
                key="explain_input",
            )
    with cols[1]:
        explain_clicked = st.button(
            "Explain", type="primary", disabled=not tx_id, use_container_width=True
        )

    items = st.session_state.get("last_scan_items", [])
    if tx_id and items:
        match = next((it for it in items if it["tx_id"] == tx_id), None)
        if match is not None:
            mc = st.columns(4)
            mc[0].metric("Score", f"{match['ensemble_score']:.3f}")
            mc[1].metric("Account", str(match["account"]))
            mc[2].metric("User", str(match["user"]))
            amount = match["debit"] if match["debit"] > 0 else match["credit"]
            mc[3].metric("Amount", f"${amount:,.2f}")
            st.markdown(
                f"**Flags:** {_render_flags(match['feature_flags'])}",
                unsafe_allow_html=True,
            )

    if explain_clicked and tx_id:
        with st.spinner("Retrieving standards and generating narrative (≈60 s on local CPU)…"):
            data = _get(f"/explain/{tx_id}?company={_selected_company}")
        if data is not None:
            if data["refused"]:
                _render_refusal(str(data["reason"]))
            n_col, c_col = st.columns([3, 2], gap="large")
            with n_col:
                st.markdown("#### Narrative")
                st.markdown(
                    f'<div class="ac-narrative">{html.escape(str(data["narrative"])).replace(chr(10), "<br/>")}</div>',
                    unsafe_allow_html=True,
                )
            with c_col:
                st.markdown("#### Citations")
                _render_citations(data["citations"])
