"""Fusion: produce a grounded risk narrative for a flagged journal entry.

This is the project's hero capability. Given:

* a flagged transaction (row from the GL plus its anomaly score and any
  feature flags), and
* a :class:`RagPipeline` over the AUASB ASA 240 corpus,

we retrieve standards passages relevant to the transaction's risk indicators
and ask the LLM to produce a short auditor-facing narrative with citations.

Safety:

* The system prompt is locked in :mod:`src.rag.prompts`.
* User-controlled fields on the transaction (``description``, ``user``,
  ``account``) are inserted into a structured transaction block, never
  concatenated into instructions.
* When retrieval confidence is below ``settings.rag_min_score`` or the LLM
  reports insufficient grounding, the explainer refuses.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.core.config import settings
from src.core.constants import REFUSAL_INSUFFICIENT_GROUNDING
from src.rag.pipeline import Citation, RagPipeline
from src.rag.prompts import ANOMALY_EXPLAIN_SYSTEM_PROMPT, ContextChunk, format_context_block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlaggedTransaction:
    """The minimal view of a flagged GL row needed by the explainer.

    Attributes:
        tx_id: Stable transaction id.
        date: Posting date as an ISO string.
        account: GL account code/name.
        debit: Debit amount.
        credit: Credit amount.
        user: User who posted the entry.
        posting_ts: Full posting timestamp as ISO string.
        description: Free-text description from the source system.
        ensemble_score: Ensemble anomaly score in ``[0, 1]``.
        feature_flags: Names of risk indicators that fired
            (e.g. ``("is_weekend", "is_round_amount")``).
    """

    tx_id: str
    date: str
    account: str
    debit: float
    credit: float
    user: str
    posting_ts: str
    description: str
    ensemble_score: float
    feature_flags: tuple[str, ...]


@dataclass(frozen=True)
class ExplainResult:
    """Result of explaining a single flagged transaction.

    Attributes:
        tx_id: The transaction id this result is for.
        narrative: Plain-English risk narrative or refusal text.
        citations: Citations corresponding to the chunks supplied to the LLM.
            Empty when the explainer refused.
        refused: True when grounding was insufficient or the LLM refused.
        reason: Short refusal reason, or empty string on success.
    """

    tx_id: str
    narrative: str
    citations: list[Citation]
    refused: bool
    reason: str


_FLAG_QUERY_TERMS: dict[str, str] = {
    "is_weekend": "weekend journal entry posting outside business days",
    "is_after_hours": "journal entries posted outside normal business hours",
    "is_round_amount": "round-dollar journal entries unusual amount",
    "is_unusual_user_account": (
        "user posting to an account outside their normal duties "
        "segregation of duties"
    ),
    "is_benford_violation": "amount distribution violation Benford's law journal entry",
    "is_benford_first_digit_9": "Benford's law first-digit nine anomalous amount",
    "is_near_duplicate": "duplicate journal entries close in time same amount",
    "is_large_amount": "high-value journal entry materiality threshold",
    "is_sensitive_account": (
        "fraud risk revenue equity manual journal entries sensitive account"
    ),
    "is_round_credit_to_revenue": "round-dollar revenue credit fictitious sales",
    "is_amount_outlier_for_account": (
        "outlier amount account population analytical procedure"
    ),
}


_LARGE_AMOUNT_THRESHOLD: float = 100_000.0
_SENSITIVE_ACCOUNTS: frozenset[str] = frozenset(
    {"4000-Revenue", "3000-Equity", "5000-COGS", "2000-AccountsPayable"}
)


def _first_digit(value: float) -> int:
    """Return the leading non-zero digit of ``|value|`` (0 when value is 0)."""
    abs_val = abs(value)
    if abs_val == 0:
        return 0
    while abs_val < 1:
        abs_val *= 10
    while abs_val >= 10:
        abs_val //= 10
    return int(abs_val)


def derive_feature_flags(
    *, posting_ts: str, debit: float, credit: float, account: str
) -> tuple[str, ...]:
    """Re-derive risk flag names from raw GL fields.

    This mirrors the booleans the anomaly feature builder computes per row,
    but only the subset that depends solely on a single row (no cross-row
    context). Used by callers that have flagged rows in CSV form and need
    the flag names back without re-running the full feature pipeline.

    Args:
        posting_ts: ISO posting timestamp.
        debit: Debit amount.
        credit: Credit amount.
        account: GL account code/name.

    Returns:
        Tuple of flag names that fired for this row.
    """
    flags: list[str] = []
    ts = datetime.fromisoformat(posting_ts)
    if ts.weekday() >= 5:
        flags.append("is_weekend")
    if ts.hour < 6 or ts.hour >= 20:
        flags.append("is_after_hours")
    amount = abs(float(debit) - float(credit))
    is_round = amount > 0 and amount % 1000 == 0
    if is_round:
        flags.append("is_round_amount")
    if _first_digit(amount) == 9:
        flags.append("is_benford_first_digit_9")
    if amount >= _LARGE_AMOUNT_THRESHOLD:
        flags.append("is_large_amount")
    if account in _SENSITIVE_ACCOUNTS:
        flags.append("is_sensitive_account")
    if is_round and float(credit) > 0 and account == "4000-Revenue":
        flags.append("is_round_credit_to_revenue")
    return tuple(flags)


def flagged_transaction_from_row(row: Mapping[str, Any]) -> FlaggedTransaction:
    """Build a :class:`FlaggedTransaction` from a flagged-CSV row.

    Args:
        row: Mapping with the columns produced by ``scripts/train_anomaly.py``
            (at minimum: ``tx_id, date, account, debit, credit, user,
            posting_ts, description, ensemble_score``).

    Returns:
        A :class:`FlaggedTransaction` with re-derived single-row feature flags.
    """
    posting_ts = str(row["posting_ts"])
    debit = float(row["debit"])
    credit = float(row["credit"])
    account = str(row["account"])
    raw_flags = row.get("feature_flags")
    if isinstance(raw_flags, str) and raw_flags.strip():
        flags = tuple(f for f in raw_flags.split(";") if f)
    else:
        flags = derive_feature_flags(
            posting_ts=posting_ts, debit=debit, credit=credit, account=account
        )
    return FlaggedTransaction(
        tx_id=str(row["tx_id"]),
        date=str(row["date"]),
        account=account,
        debit=debit,
        credit=credit,
        user=str(row["user"]),
        posting_ts=posting_ts,
        description=str(row["description"]),
        ensemble_score=float(row["ensemble_score"]),
        feature_flags=flags,
    )


def build_query(tx: FlaggedTransaction) -> str:
    """Build a retrieval query string from a flagged transaction's signals.

    The query always anchors on ASA 240 fraud-risk journal-entry testing and
    adds risk-specific phrases for each active feature flag. When no flags
    fired, the query falls back to the generic ASA 240 frame plus the
    account name.

    Args:
        tx: The flagged transaction.

    Returns:
        A short retrieval query string suitable for hybrid retrieval.
    """
    parts: list[str] = [
        "ASA 240 auditor responsibilities fraud risk journal entry testing",
    ]
    for flag in tx.feature_flags:
        term = _FLAG_QUERY_TERMS.get(flag)
        if term:
            parts.append(term)
    if len(parts) == 1:
        # No specific flags fired; broaden with the account label.
        parts.append(f"unusual posting to account {tx.account}")
    return " ".join(parts)


def format_transaction_block(tx: FlaggedTransaction) -> str:
    """Render the transaction as a structured block for the prompt.

    Args:
        tx: The flagged transaction.

    Returns:
        A multi-line string with one ``label: value`` per line. User-supplied
        free text is included as a quoted field so the LLM cannot mistake it
        for an instruction.
    """
    amount = tx.debit if tx.debit > 0 else tx.credit
    side = "debit" if tx.debit > 0 else "credit"
    flags = ", ".join(tx.feature_flags) if tx.feature_flags else "(none)"
    # Strip newlines from free-text fields so they can't break the structure.
    safe_description = tx.description.replace("\n", " ").replace("\r", " ")
    safe_user = tx.user.replace("\n", " ").replace("\r", " ")
    safe_account = tx.account.replace("\n", " ").replace("\r", " ")
    return (
        "TRANSACTION:\n"
        f"  tx_id: {tx.tx_id}\n"
        f"  date: {tx.date}\n"
        f"  posting_ts: {tx.posting_ts}\n"
        f"  account: {safe_account}\n"
        f"  amount: {amount:.2f} ({side})\n"
        f"  user: {safe_user}\n"
        f'  source_ref: "{safe_description}"\n'
        f"  ensemble_anomaly_score: {tx.ensemble_score:.3f}\n"
        f"  risk_indicators: {flags}"
    )


def _format_explain_prompt(tx: FlaggedTransaction, chunks: list[ContextChunk]) -> str:
    """Build the user-role prompt for the explainer."""
    context_block = format_context_block(chunks)
    transaction_block = format_transaction_block(tx)
    return (
        "CONTEXT:\n"
        f"{context_block}\n\n"
        f"{transaction_block}\n\n"
        "TASK:\n"
        "Explain in 3-5 sentences why this transaction is risky from an "
        "auditor's perspective. Reference the risk indicators listed above "
        "and cite supporting standards passages by their [n] tag. If the "
        "CONTEXT does not support a grounded explanation, reply exactly: "
        "INSUFFICIENT_GROUNDING."
    )


class AnomalyExplainer:
    """Produces grounded narratives for flagged transactions.

    Attributes:
        pipeline: A configured :class:`RagPipeline` used for hybrid retrieval
            and LLM access. The pipeline's ``index``, ``llm`` and BM25 corpus
            are reused as-is.
    """

    def __init__(self, *, pipeline: RagPipeline) -> None:
        """Construct the explainer.

        Args:
            pipeline: The RAG pipeline whose retrieval + LLM are reused.
        """
        self.pipeline = pipeline

    def explain(self, tx: FlaggedTransaction, *, company: str | None = None) -> ExplainResult:
        """Generate a grounded narrative for one flagged transaction.

        Args:
            tx: The flagged transaction.
            company: Optional ticker; when set, retrieval is restricted to
                ``{"AUASB", "ASX-<ticker>"}``.

        Returns:
            :class:`ExplainResult` containing the narrative + citations, or a
            refusal when grounding is insufficient.
        """
        query = build_query(tx)
        k = settings.rag_top_k
        hits = self.pipeline.retrieve(query, k=k, company=company)
        if not hits:
            logger.warning("no retrieval hits for transaction", extra={"tx_id": tx.tx_id})
            return ExplainResult(
                tx_id=tx.tx_id,
                narrative=REFUSAL_INSUFFICIENT_GROUNDING,
                citations=[],
                refused=True,
                reason="no retrieval hits",
            )

        top_score = hits[0].score
        if top_score < settings.rag_min_score:
            logger.warning(
                "grounding below threshold",
                extra={"tx_id": tx.tx_id, "top_score": top_score},
            )
            return ExplainResult(
                tx_id=tx.tx_id,
                narrative=REFUSAL_INSUFFICIENT_GROUNDING,
                citations=[],
                refused=True,
                reason="grounding below threshold",
            )

        chunks = [
            ContextChunk(
                tag=i + 1,
                source=hit.source,
                section=hit.section,
                page=hit.page,
                text=hit.text,
            )
            for i, hit in enumerate(hits)
        ]
        prompt = _format_explain_prompt(tx, chunks)
        narrative = self.pipeline.llm.complete(
            system=ANOMALY_EXPLAIN_SYSTEM_PROMPT,
            user=prompt,
        )

        if narrative.strip() == "INSUFFICIENT_GROUNDING":
            return ExplainResult(
                tx_id=tx.tx_id,
                narrative=REFUSAL_INSUFFICIENT_GROUNDING,
                citations=[],
                refused=True,
                reason="llm reported insufficient grounding",
            )

        citations = [
            Citation(
                tag=i + 1,
                source=hit.source,
                section=hit.section,
                page=hit.page,
                chunk_id=hit.chunk_id,
            )
            for i, hit in enumerate(hits)
        ]
        return ExplainResult(
            tx_id=tx.tx_id,
            narrative=narrative,
            citations=citations,
            refused=False,
            reason="",
        )
