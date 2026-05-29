"""VectorMemory — semantic retrieval over a Qdrant collection.

On add, messages are embedded and upserted (and also kept in the session store for
durable, ordered recall). ``aget_context(query=...)`` returns the messages most relevant
to the query, ordered chronologically so the model reads them in natural order.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import EmbedMode, Message, RetrievalResult, embed_texts
from ..embeddings.base import Embedder
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.session_store import SessionStore


class VectorMemory(BaseMemory):
    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        *,
        top_k: int = 5,
        embed_mode: EmbedMode = "per_message",
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._embedder = embedder
        self._vectors = vector_store
        self._top_k = top_k
        self._embed_mode = embed_mode

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        msgs = self._coerce(messages)
        if not msgs:
            return
        vectors = await self._embedder.aembed(embed_texts(msgs, self._embed_mode))
        await self._store.append(self.session_id, msgs)
        await self._vectors.upsert(msgs, vectors)

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        vector = await self._embedder.aembed_one(query)
        return await self._vectors.search(self.session_id, vector, k=k)

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        if query is None:
            # Without a query there is nothing to rank against; fall back to recency.
            return await self._store.recent(self.session_id, limit or self._top_k)
        results = await self.asearch(query, k=limit or self._top_k)
        # Return in chronological order for natural reading by the model.
        return [r.message for r in sorted(results, key=lambda r: r.message.timestamp)]

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
        await self._vectors.clear(self.session_id)
