"""End-to-end RAG pipeline: guardrails -> hybrid retrieval -> generation.

* Dense scores come from :class:`ChromaIndex`.
* Lexical scores come from BM25 over the same chunk corpus.
* The two rankings are fused with reciprocal-rank fusion (RRF).
* Results below ``settings.rag_min_score`` (dense) trigger a refusal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.core.constants import REFUSAL_INSUFFICIENT_GROUNDING, RRF_K
from src.llm.client import LLMClient
from src.rag.guardrails import check_question
from src.rag.indexer import ChromaIndex, RetrievedChunk
from src.rag.prompts import RAG_SYSTEM_PROMPT, ContextChunk, format_user_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Citation:
    """A single citation returned alongside a generated answer.

    Attributes:
        tag: 1-based tag matching the ``[n]`` reference in the answer text.
        source: Source label of the cited chunk.
        section: Section heading of the cited chunk.
        page: Page number of the cited chunk, when known.
        chunk_id: Stable id of the cited chunk.
    """

    tag: int
    source: str
    section: str
    page: int | None
    chunk_id: str


@dataclass(frozen=True)
class AnswerResult:
    """Result of running the pipeline against a user question.

    Attributes:
        answer: Generated answer text, or refusal message.
        citations: Citations corresponding to the chunks supplied to the LLM.
            Empty list when the pipeline refused.
        refused: True when guardrails or grounding caused a refusal.
        reason: Short refusal reason, or empty string on success.
    """

    answer: str
    citations: list[Citation]
    refused: bool
    reason: str


def _tokenize(text: str) -> list[str]:
    """Lowercase, whitespace-tokenize a string for BM25.

    Args:
        text: Input text.

    Returns:
        Lowercase word tokens.
    """
    return text.lower().split()


def _rrf_fuse(
    dense: list[RetrievedChunk],
    lexical_ids_ranked: list[str],
    *,
    k: int,
) -> list[RetrievedChunk]:
    """Combine dense and lexical rankings with reciprocal-rank fusion.

    Args:
        dense: Dense hits in score order.
        lexical_ids_ranked: Chunk ids ranked by BM25 score.
        k: Maximum results to return after fusion.

    Returns:
        Up to ``k`` chunks ordered by fused score, drawn from the dense pool.
    """
    by_id: dict[str, RetrievedChunk] = {hit.chunk_id: hit for hit in dense}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(dense):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, chunk_id in enumerate(lexical_ids_ranked):
        if chunk_id in by_id:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [by_id[cid] for cid, _ in ordered if cid in by_id][:k]


class RagPipeline:
    """Coordinates guardrails, retrieval, and LLM generation.

    Attributes:
        index: Dense vector index.
        llm: LLM client used for generation.
        bm25_corpus: Tokenised documents in the same order as ``bm25_ids``.
        bm25_ids: Chunk ids parallel to ``bm25_corpus``.
        bm25: Pre-built BM25 model over the corpus, or ``None`` until built.
    """

    def __init__(
        self,
        *,
        index: ChromaIndex,
        llm: LLMClient,
        bm25_documents: list[tuple[str, str]] | None = None,
    ) -> None:
        """Construct the pipeline.

        Args:
            index: Dense vector index.
            llm: LLM client used for generation.
            bm25_documents: Optional ``(chunk_id, text)`` pairs used to build
                the lexical index. When omitted, lexical retrieval is skipped
                and the pipeline returns the dense ranking unchanged.
        """
        self.index = index
        self.llm = llm
        self.bm25_ids: list[str] = []
        self.bm25_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        if bm25_documents:
            self.bm25_ids = [doc_id for doc_id, _ in bm25_documents]
            self.bm25_corpus = [_tokenize(text) for _, text in bm25_documents]
            self.bm25 = BM25Okapi(self.bm25_corpus)

    def _lexical_rank(self, query: str, k: int) -> list[str]:
        """Return chunk ids ranked by BM25 score.

        Args:
            query: Tokenised query string.
            k: Maximum ids to return.

        Returns:
            Ranked chunk ids, highest score first. Empty when BM25 was not
            built.
        """
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.bm25_ids[i] for i in ranked[:k]]

    def retrieve(self, query: str, *, k: int) -> list[RetrievedChunk]:
        """Run hybrid retrieval (dense + BM25 + RRF) without generation.

        Args:
            query: Already-sanitised retrieval query string.
            k: Maximum fused results to return.

        Returns:
            Up to ``k`` chunks ordered by fused score; empty when the index
            returns no dense hits.
        """
        dense_hits = self.index.query(query, k=k * 2)
        if not dense_hits:
            return []
        lexical_ids = self._lexical_rank(query, k=k * 2)
        return _rrf_fuse(dense_hits, lexical_ids, k=k)

    def answer(self, question: str) -> AnswerResult:
        """Run the full pipeline for one question.

        Args:
            question: Raw user question.

        Returns:
            :class:`AnswerResult` with either a grounded answer + citations
            or a refusal.
        """
        guard = check_question(question)
        if not guard.ok:
            logger.warning("guardrail refusal", extra={"reason": guard.reason})
            return AnswerResult(
                answer=REFUSAL_INSUFFICIENT_GROUNDING,
                citations=[],
                refused=True,
                reason=guard.reason,
            )

        sanitized = guard.sanitized
        k = settings.rag_top_k
        dense_hits = self.index.query(sanitized, k=k * 2)
        if not dense_hits:
            logger.warning("no dense hits", extra={"k": k})
            return AnswerResult(
                answer=REFUSAL_INSUFFICIENT_GROUNDING,
                citations=[],
                refused=True,
                reason="no retrieval hits",
            )

        top_dense_score = dense_hits[0].score
        if top_dense_score < settings.rag_min_score:
            logger.warning(
                "grounding below threshold",
                extra={"top_score": top_dense_score, "threshold": settings.rag_min_score},
            )
            return AnswerResult(
                answer=REFUSAL_INSUFFICIENT_GROUNDING,
                citations=[],
                refused=True,
                reason="grounding below threshold",
            )

        lexical_ids = self._lexical_rank(sanitized, k=k * 2)
        fused = _rrf_fuse(dense_hits, lexical_ids, k=k)

        context_chunks = [
            ContextChunk(
                tag=i + 1,
                source=hit.source,
                section=hit.section,
                page=hit.page,
                text=hit.text,
            )
            for i, hit in enumerate(fused)
        ]
        prompt = format_user_prompt(sanitized, context_chunks)
        answer_text = self.llm.complete(system=RAG_SYSTEM_PROMPT, user=prompt)

        if answer_text.strip() == "INSUFFICIENT_GROUNDING":
            return AnswerResult(
                answer=REFUSAL_INSUFFICIENT_GROUNDING,
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
            for i, hit in enumerate(fused)
        ]
        return AnswerResult(answer=answer_text, citations=citations, refused=False, reason="")
