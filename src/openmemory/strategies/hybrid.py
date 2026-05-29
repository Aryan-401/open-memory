"""HybridMemory — recency window unioned with semantic retrieval.

The recommended default for chat assistants: it always includes the most recent turns
(so the model has immediate continuity) plus the turns most relevant to the current
query pulled from anywhere in history (so older-but-pertinent context resurfaces).
Results are deduplicated by message id and returned in chronological order.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import EmbedMode, Message, RetrievalResult
from ..embeddings.base import Embedder
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.session_store import SessionStore
from .vector import VectorMemory
from .window import WindowMemory


class HybridMemory(BaseMemory):
    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        *,
        window_size: int = 10,
        top_k: int = 5,
        token_budget: int | None = None,
        model: str = "gpt-4o-mini",
        embed_mode: EmbedMode = "per_message",
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._window = WindowMemory(
            session_id, store, n=window_size, token_budget=token_budget, model=model
        )
        self._vector = VectorMemory(
            session_id, store, embedder, vector_store, top_k=top_k, embed_mode=embed_mode
        )
        self._top_k = top_k

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        # Delegate to VectorMemory which writes both the session store and the index.
        await self._vector.aadd(messages)

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return await self._vector.asearch(query, k=k)

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        recent = await self._window.aget_context(limit=limit, token_budget=token_budget)

        retrieved: list[Message] = []
        if query is not None:
            results = await self._vector.asearch(query, k=self._top_k)
            recent_ids = {m.id for m in recent}
            retrieved = [r.message for r in results if r.message.id not in recent_ids]

        # Merge, dedupe, and order chronologically.
        merged: dict[str, Message] = {}
        for msg in retrieved + recent:
            merged[msg.id] = msg
        return sorted(merged.values(), key=lambda m: m.timestamp)

    async def aclear(self) -> None:
        await self._vector.aclear()
