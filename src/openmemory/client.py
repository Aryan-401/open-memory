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
from .strategies.graph import GraphMemory, NetworkxGraphStore
from .strategies.hierarchical import HierarchicalMemory
from .strategies.hybrid import HybridMemory
from .strategies.summary import SummaryMemory
from .strategies.vector import VectorMemory
from .strategies.window import WindowMemory

Strategy = Literal[
    "buffer", "window", "vector", "hybrid", "hierarchical", "summary", "facts", "graph"
]


def _without(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return a copy of ``d`` with the given keys removed."""
    return {k: v for k, v in d.items() if k not in keys}


class OpenMemory:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._store: SessionStore | None = None
        self._embedder: Embedder | None = None
        self._llm: LLM | None = None
        self._vectors: QdrantVectorStore | None = None
        self._fact_vectors: QdrantVectorStore | None = None
        self._summarizer: Summarizer | None = None
        self._nx_graph: NetworkxGraphStore | None = None

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

    def _fact_vector_store(self) -> QdrantVectorStore:
        """Separate Qdrant collection for extracted facts (avoids mixing with raw messages)."""
        if self._fact_vectors is None:
            self._fact_vectors = QdrantVectorStore(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                collection=f"{self.config.qdrant_collection}_facts",
            )
        return self._fact_vectors

    def _summarizer_(self) -> Summarizer:
        if self._summarizer is None:
            self._summarizer = Summarizer(self._llm_())
        return self._summarizer

    def _nx_graph_store(self) -> NetworkxGraphStore:
        if self._nx_graph is None:
            self._nx_graph = NetworkxGraphStore()
        return self._nx_graph

    def _strategy_llm(
        self, provider_override: str | None, model_override: str | None
    ) -> LLM:
        """Build an LLM for a specific strategy, reusing the shared one when possible."""
        cfg = self.config
        effective_provider = provider_override or cfg.llm_provider
        effective_model = model_override or cfg.llm_model
        if effective_provider == cfg.llm_provider and effective_model == cfg.llm_model:
            return self._llm_()
        # Different provider/model — build a dedicated instance.
        override_cfg = cfg.model_copy(
            update={"llm_provider": effective_provider, "llm_model": effective_model}
        )
        return build_llm(override_cfg)

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
            llm = self._strategy_llm(cfg.summary_llm_provider, cfg.summary_llm_model)
            summarizer = Summarizer(llm)
            params = {"buffer_size": cfg.summary_buffer_size}
            params.update(_without(overrides, "embed_mode"))
            return SummaryMemory(session_id, store, summarizer, **params)

        if strategy == "facts":
            llm = self._strategy_llm(cfg.facts_llm_provider, cfg.facts_llm_model)
            params = {
                "top_k": cfg.retrieval_top_k,
                "dedup_threshold": cfg.facts_dedup_threshold,
            }
            params.update(_without(overrides, "embed_mode"))
            return FactExtractionMemory(
                session_id, store, llm, self._embedder_(), self._fact_vector_store(),
                **params,
            )

        if strategy == "graph":
            llm = self._strategy_llm(cfg.graph_llm_provider, cfg.graph_llm_model)
            params = {"hops": cfg.graph_hops}
            params.update(_without(overrides, "embed_mode"))
            return GraphMemory(session_id, store, llm, self._nx_graph_store(), **params)

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
        import asyncio

        # When asyncio.run() shuts down the loop, any in-flight SSL/TLS connections
        # (from Qdrant or the LiteLLM HTTP client) may fire:
        #   "Fatal error on SSL transport … OSError: Bad file descriptor"
        #   "RuntimeError: Event loop is closed"
        # through asyncio's own exception handler.  These are cosmetic — the program
        # has already finished successfully — but confusing.  Install a one-time quiet
        # handler *before* closing connections so it is in place when teardown fires.
        try:
            loop = asyncio.get_running_loop()
            _orig = loop.get_exception_handler()

            def _quiet(lp: asyncio.AbstractEventLoop, ctx: dict) -> None:
                exc = ctx.get("exception")
                if isinstance(exc, (RuntimeError, OSError)):
                    msg = str(exc)
                    if "Event loop is closed" in msg or "Bad file descriptor" in msg:
                        return
                if _orig is not None:
                    _orig(lp, ctx)
                else:
                    lp.default_exception_handler(ctx)

            loop.set_exception_handler(_quiet)
        except RuntimeError:
            pass

        if self._vectors is not None:
            await self._vectors.close()
        if self._fact_vectors is not None:
            await self._fact_vectors.close()
        if isinstance(self._store, SQLiteSessionStore):
            await self._store.close()
        # Give any remaining I/O callbacks one loop iteration to drain.
        await asyncio.sleep(0)
