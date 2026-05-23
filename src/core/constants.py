"""Static constants — values that are not user-tunable via environment."""

from __future__ import annotations

CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64
EMBED_DIM: int = 384

RRF_K: int = 60

REFUSAL_INSUFFICIENT_GROUNDING: str = "insufficient grounding — manual review required"
