"""Central configuration for open-memory.

All settings can be supplied programmatically (``Config(...)``) or via environment
variables prefixed with ``OPENMEMORY_`` (e.g. ``OPENMEMORY_QDRANT_URL``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedderProvider = Literal["openai", "litellm", "local"]
LLMProvider = Literal["openai", "litellm", "local"]
SessionStoreKind = Literal["memory", "sqlite"]
EmbedMode = Literal["per_message", "paired"]


class Config(BaseSettings):
    """Runtime configuration shared by the :class:`~openmemory.client.OpenMemory` facade."""

    model_config = SettingsConfigDict(
        env_prefix="OPENMEMORY_",
        env_file=".env",
        extra="ignore",
    )

    # --- Embeddings ---
    embedder_provider: EmbedderProvider = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int | None = Field(
        default=None,
        description="Vector size. If None, inferred lazily from the first embedding call.",
    )

    # --- Chat / summarization LLM ---
    llm_provider: LLMProvider = "openai"
    llm_model: str = "gpt-4o-mini"

    # --- OpenAI-compatible client (used by openai providers) ---
    openai_api_key: str | None = None
    openai_base_url: str | None = Field(
        default=None,
        description="Override for any OpenAI-compatible endpoint (vLLM, Ollama, etc.).",
    )

    # --- Local (huggingface / sentence-transformers) ---
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Qdrant ---
    qdrant_url: str | None = Field(
        default=None,
        description="e.g. http://localhost:6333. If None, an in-memory Qdrant is used.",
    )
    qdrant_api_key: str | None = None
    qdrant_collection: str = "openmemory"

    # --- Session persistence ---
    session_store: SessionStoreKind = "memory"
    sqlite_path: str = "openmemory.db"

    # --- Strategy defaults ---
    window_size: int = 20
    token_budget: int = 4000
    working_context_tokens: int = 2000
    retrieval_top_k: int = 5
    embed_mode: EmbedMode = "per_message"

    # --- SummaryMemory ---
    summary_llm_provider: str | None = None  # falls back to llm_provider
    summary_llm_model: str | None = None     # falls back to llm_model
    summary_buffer_size: int = 6             # recent messages kept verbatim

    # --- FactExtractionMemory ---
    facts_llm_provider: str | None = None   # falls back to llm_provider
    facts_llm_model: str | None = None      # falls back to llm_model
    facts_dedup_threshold: float = 0.85     # cosine similarity above which a fact is a duplicate

    # --- GraphMemory ---
    graph_llm_provider: str | None = None   # falls back to llm_provider
    graph_llm_model: str | None = None      # falls back to llm_model
    graph_hops: int = 1                     # neighborhood traversal depth
