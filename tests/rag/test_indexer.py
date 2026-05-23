"""Tests for the ChromaIndex wrapper using fake embedder + fake collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.rag.indexer import ChromaIndex, RetrievedChunk
from src.rag.ingest import Chunk


class _FakeEmbedder:
    def encode(self, texts: list[str], **_: Any) -> Any:
        # 4-dim deterministic embedding based on text length
        return np.array([[len(t), len(t) % 2, len(t) % 3, len(t) % 5] for t in texts], dtype=float)


class _FakeCollection:
    def __init__(self) -> None:
        self.upserted: dict[str, Any] = {}
        self.next_query: dict[str, Any] | None = None
        self.last_query_kwargs: dict[str, Any] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for i, cid in enumerate(ids):
            self.upserted[cid] = {
                "embedding": embeddings[i],
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def query(
        self, *, query_embeddings: list[list[float]], n_results: int, **kwargs: Any
    ) -> dict[str, Any]:
        assert query_embeddings  # used
        self.last_query_kwargs = kwargs
        if self.next_query is not None:
            return self.next_query
        ids = list(self.upserted.keys())[:n_results]
        docs = [self.upserted[i]["document"] for i in ids]
        metas = [self.upserted[i]["metadata"] for i in ids]
        distances = [0.1 * (rank + 1) for rank in range(len(ids))]
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
        }

    def count(self) -> int:
        return len(self.upserted)


def test_add_and_query_roundtrip() -> None:
    collection = _FakeCollection()
    index = ChromaIndex(collection=collection, embedder=_FakeEmbedder())

    chunks = [
        Chunk(chunk_id="x:1:0", source="AUASB", section="Section 1", page=1, text="alpha"),
        Chunk(chunk_id="x:2:0", source="AUASB", section="Section 2", page=2, text="beta beta"),
    ]
    index.add(chunks)

    assert index.count() == 2

    hits = index.query("query", k=2)
    assert len(hits) == 2
    assert isinstance(hits[0], RetrievedChunk)
    assert hits[0].score >= hits[1].score
    assert 0.0 <= hits[1].score <= 1.0
    assert hits[0].source == "AUASB"
    assert hits[0].page == 1


def test_add_empty_is_noop() -> None:
    collection = _FakeCollection()
    index = ChromaIndex(collection=collection, embedder=_FakeEmbedder())
    index.add([])
    assert index.count() == 0


def test_query_passes_where_filter_through() -> None:
    collection = _FakeCollection()
    index = ChromaIndex(collection=collection, embedder=_FakeEmbedder())
    index.add(
        [
            Chunk(chunk_id="x:1:0", source="AUASB", section="s", page=1, text="alpha"),
        ]
    )

    filter_ = {"source": {"$in": ["AUASB", "ASX-WOW"]}}
    index.query("query", k=2, where=filter_)
    assert collection.last_query_kwargs.get("where") == filter_

    index.query("query", k=2)
    assert "where" not in collection.last_query_kwargs
