"""ChromaDB persistent index over chunked corpus.

Wraps the embedded ``PersistentClient`` plus a SentenceTransformer embedder.
Tests substitute a fake collection through the constructor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.core.config import settings
from src.rag.ingest import Chunk

logger = logging.getLogger(__name__)


class _Embedder(Protocol):
    """Minimal embedding-model interface used by the indexer."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """Return embeddings for the given texts."""
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieval hit returned by :meth:`ChromaIndex.query`.

    Attributes:
        chunk_id: Stable id matching the original :class:`Chunk`.
        source: Source label.
        section: Section heading.
        page: Page number when known.
        text: Chunk text.
        score: Dense similarity score in [0, 1]. Higher is better.
    """

    chunk_id: str
    source: str
    section: str
    page: int | None
    text: str
    score: float


def _load_embedder(model_name: str) -> _Embedder:
    """Import and instantiate the SentenceTransformer model.

    Args:
        model_name: HuggingFace model id.

    Returns:
        A loaded :class:`sentence_transformers.SentenceTransformer`.
    """
    from sentence_transformers import SentenceTransformer

    model: _Embedder = SentenceTransformer(model_name)
    return model


def _open_collection(chroma_dir: Path, collection_name: str) -> Any:
    """Open or create the persistent Chroma collection.

    The collection is configured with cosine distance so that the
    ``distance`` field returned by Chroma is ``1 - cosine_similarity`` and
    ``score = 1 - distance`` lands in ``[0, 1]``.

    Args:
        chroma_dir: Filesystem path for the persistent store.
        collection_name: Logical collection name.

    Returns:
        A Chroma ``Collection`` instance.
    """
    import chromadb

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


class ChromaIndex:
    """Thin wrapper around a Chroma collection with our chunk metadata schema.

    Attributes:
        collection: The underlying Chroma collection (or a test double).
        embedder: Embedding model used for both upserts and queries.
    """

    def __init__(
        self,
        *,
        collection: Any | None = None,
        embedder: _Embedder | None = None,
    ) -> None:
        """Construct the index.

        Args:
            collection: Optional pre-built Chroma collection (used by tests).
                When omitted, opens the default persistent collection from
                settings.
            embedder: Optional embedding model (used by tests). When omitted,
                loads the model configured in settings.
        """
        self.collection = collection or _open_collection(
            settings.chroma_dir, settings.chroma_collection
        )
        self.embedder = embedder or _load_embedder(settings.embedding_model)

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and upsert a batch of chunks.

        Args:
            chunks: Chunks to index. No-op when the list is empty.
        """
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).tolist()
        metadatas = [{"source": c.source, "section": c.section, "page": c.page} for c in chunks]
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("chunks upserted", extra={"count": len(chunks)})

    def query(self, text: str, *, k: int) -> list[RetrievedChunk]:
        """Run a dense similarity search.

        Args:
            text: Query string.
            k: Maximum hits to return.

        Returns:
            Up to ``k`` :class:`RetrievedChunk` results, ordered by score
            (highest first). Score is ``1 - distance`` clipped to ``[0, 1]``.
        """
        embedding = self.embedder.encode([text], convert_to_numpy=True).tolist()
        raw = self.collection.query(query_embeddings=embedding, n_results=k)
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        out: list[RetrievedChunk] = []
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            page_raw = meta.get("page") if isinstance(meta, dict) else None
            page = int(page_raw) if isinstance(page_raw, int) else None
            out.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    source=str(meta.get("source", "")) if isinstance(meta, dict) else "",
                    section=str(meta.get("section", "")) if isinstance(meta, dict) else "",
                    page=page,
                    text=str(doc),
                    score=score,
                )
            )
        return out

    def count(self) -> int:
        """Return the number of chunks currently in the collection.

        Returns:
            Chunk count.
        """
        result = self.collection.count()
        return int(result)
