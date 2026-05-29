"""The ``OpenMemory`` facade: pick a strategy, get a session, start storing context.

Shared, expensive resources (embedder, LLM, vector store, summarizer) are built lazily
the first time a strategy needs them — so the naive ``buffer``/``window`` strategies work
with zero configuration and no API keys.
"""

from __future__ import annotations

from typing import Any, Literal

from .config import Config
from .core.base import BaseMemory
from .core.models import EmbedMode, embed_texts
from .embeddings.base import Embedder, build_embedder
from .llm.base import LLM, build_llm
from .llm.summarizer import Summarizer
from .session import Session
from .storage.qdrant_store import QdrantVectorStore
from .storage.session_store import InMemorySessionStore, SessionStore
from .storage.sqlite_store import SQLiteSessionStore
from .strategies.buffer import BufferMemory
from .strategies.facts import FactExtractionMemory
from .strategies.graph import GraphMemory
from .strategies.hierarchical import HierarchicalMemory
from .strategies.hybrid import HybridMemory
from .strategies.summary import SummaryMemory
from .strategies.vector import VectorMemory
from .strategies.window import WindowMemory

Strategy = Literal[
    "buffer", "window", "vector", "hybrid", "hierarchical", "summary", "facts", "graph"
]


class OpenMemory:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._store: SessionStore | None = None
        self._embedder: Embedder | None = None
        self._llm: LLM | None = None
        self._vectors: QdrantVectorStore | None = None
        self._summarizer: Summarizer | None = None

    # --- Lazy shared resources ---

    def _session_store(self) -> SessionStore:
        if self._store is None:
            if self.config.session_store == "sqlite":
                self._store = SQLiteSessionStore(self.config.sqlite_path)
            else:
                self._store = InMemorySessionStore()
        return self._store

    def _embedder_(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.config)
        return self._embedder

    def _llm_(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.config)
        return self._llm

    def _vector_store(self) -> QdrantVectorStore:
        if self._vectors is None:
            self._vectors = QdrantVectorStore(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                collection=self.config.qdrant_collection,
            )
        return self._vectors

    def _summarizer_(self) -> Summarizer:
        if self._summarizer is None:
            self._summarizer = Summarizer(self._llm_())
        return self._summarizer

    # --- Session factory ---

    def session(
        self,
        session_id: str,
        *,
        strategy: Strategy = "buffer",
        metadata: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> Session:
        """Create or resume a session backed by ``strategy``.

        ``overrides`` are passed to the strategy constructor (e.g. ``n=50`` for window,
        ``window_size``/``top_k`` for hybrid), overriding config defaults.
        """
        memory = self._build_strategy(session_id, strategy, overrides)
        return Session(session_id, memory, metadata=metadata)

    def _build_strategy(
        self, session_id: str, strategy: Strategy, overrides: dict[str, Any]
    ) -> BaseMemory:
        store = self._session_store()
        cfg = self.config

        if strategy == "buffer":
            return BufferMemory(session_id, store)
        if strategy == "window":
            params: dict[str, Any] = {
                "n": cfg.window_size,
                "token_budget": None,
                "model": cfg.llm_model,
            }
            params.update(overrides)
            return WindowMemory(session_id, store, **params)
        if strategy == "vector":
            params = {"top_k": cfg.retrieval_top_k, "embed_mode": cfg.embed_mode}
            params.update(overrides)
            return VectorMemory(
                session_id, store, self._embedder_(), self._vector_store(), **params
            )
        if strategy == "hybrid":
            params = {
                "window_size": cfg.window_size,
                "top_k": cfg.retrieval_top_k,
                "model": cfg.llm_model,
                "embed_mode": cfg.embed_mode,
            }
            params.update(overrides)
            return HybridMemory(
                session_id, store, self._embedder_(), self._vector_store(), **params
            )
        if strategy == "hierarchical":
            params = {
                "working_context_tokens": cfg.working_context_tokens,
                "top_k": cfg.retrieval_top_k,
                "model": cfg.llm_model,
                "embed_mode": cfg.embed_mode,
            }
            params.update(overrides)
            return HierarchicalMemory(
                session_id,
                store,
                self._embedder_(),
                self._vector_store(),
                self._summarizer_(),
                **params,
            )
        if strategy == "summary":
            return SummaryMemory(session_id)
        if strategy == "facts":
            return FactExtractionMemory(session_id)
        if strategy == "graph":
            return GraphMemory(session_id)
        raise ValueError(f"Unknown strategy: {strategy!r}")

    async def areindex(
        self, session_id: str, *, embed_mode: EmbedMode | None = None
    ) -> int:
        """Embed all stored messages for a session and upsert them into the vector store.

        Useful when turns were added under a non-semantic strategy (``buffer``/``window``)
        and you later switch to a semantic one — call this so those turns become
        retrievable. Idempotent: upserts are keyed by message id.

        ``embed_mode`` overrides the config default for this reindex run.
        """
        messages = await self._session_store().get_all(session_id)
        if not messages:
            return 0
        mode = embed_mode or self.config.embed_mode
        vectors = await self._embedder_().aembed(embed_texts(messages, mode))
        await self._vector_store().upsert(messages, vectors)
        return len(messages)

    async def aclose(self) -> None:
        """Release shared resources (vector-store and SQLite connections)."""
        if self._vectors is not None:
            await self._vectors.close()
        if isinstance(self._store, SQLiteSessionStore):
            await self._store.close()
