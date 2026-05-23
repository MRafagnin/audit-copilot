"""Application settings loaded from environment / .env.

All modules import the singleton `settings` from this file. No other module
should call `os.getenv` directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for AuditCopilot.

    Attributes:
        ollama_base_url: Base URL of the local Ollama HTTP server.
        llm_model: Ollama model tag used for generation.
        llm_timeout_seconds: Per-request timeout for LLM calls.
        embedding_model: HuggingFace model id for sentence-transformers embeddings.
        chroma_dir: Filesystem path for the embedded ChromaDB persistent store.
        chroma_collection: Name of the Chroma collection holding corpus chunks.
        rag_top_k: Number of chunks returned after hybrid fusion.
        rag_min_score: Minimum top fused score below which the pipeline refuses.
        http_user_agent: Identifier sent on outbound HTTP requests when fetching
            public PDFs (AUASB standards, ASX annual reports). Polite scraping.
        demo_ticker: ASX ticker of the company whose annual report grounds the
            RAG corpus (default ``WOW`` — Woolworths Group).
        anomaly_seed: Random seed for synthetic GL generation and model training.
        gl_row_count: Number of journal-entry rows in the synthetic dataset.
        log_level: Root log level (DEBUG / INFO / WARNING / ERROR).
        log_format: ``json`` for structured logs, ``text`` for human-readable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_timeout_seconds: int = 120

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "audit_corpus"

    rag_top_k: int = 8
    rag_min_score: float = 0.35

    http_user_agent: str = Field(default="AuditCopilot dev@example.com")
    demo_ticker: str = "WOW"

    anomaly_seed: int = 42
    gl_row_count: int = 50_000

    log_level: str = "INFO"
    log_format: str = "json"


settings = Settings()
