"""SparseHybridMemory — BM25 lexical retrieval fused with dense vector retrieval via RRF.

Unlike :class:`~openmemory.strategies.hybrid.HybridMemory` (which unions a *recency
window* with semantic ANN), ``SparseHybridMemory`` fuses two *retrieval* signals:

- **BM25** (keyword precision): exact and near-exact term matching; great for names,
  codes, domain-specific vocabulary, and queries where the user reuses their own phrasing.
- **Dense vector ANN** (semantic recall): catches paraphrases and topical similarity even
  when the user's words don't match the stored text verbatim.

The two ranked lists are merged with Reciprocal Rank Fusion (RRF): each document earns
``1 / (k + rank)`` per list, summed across lists. This sidesteps the need to normalise
BM25 scores (arbitrary float) against cosine sims (bounded [-1, 1]).

There is no recency-window bias: every message in the session is a retrieval candidate.
This is the right default when histories are large and keyword precision matters.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import EmbedMode, Message, RetrievalResult
from ..embeddings.base import Embedder
from ..storage.bm25_store import BM25Store, rrf
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.session_store import SessionStore
from .vector import VectorMemory


class SparseHybridMemory(BaseMemory):
    """BM25 + dense vector retrieval fused via Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    top_k:
        Number of candidates fetched from each retrieval channel (BM25 and vector)
        before fusion. The fused list is also capped at ``top_k``.
    embed_mode:
        ``"per_message"`` or ``"paired"`` — forwarded to the underlying
        :class:`~openmemory.strategies.vector.VectorMemory`.
    rrf_k:
        RRF rank-smoothing constant. ``k=60`` is the standard default from the original
        RRF paper. Higher k → flatter rank differences; lower k → winner-takes-all.
    """

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        bm25_store: BM25Store,
        *,
        top_k: int = 5,
        embed_mode: EmbedMode = "per_message",
        rrf_k: int = 60,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._vector = VectorMemory(
            session_id, store, embedder, vector_store, top_k=top_k, embed_mode=embed_mode
        )
        self._bm25 = bm25_store
        self._top_k = top_k
        self._rrf_k = rrf_k

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        msgs = self._coerce(messages)
        if not msgs:
            return
        # VectorMemory handles session-store append + embedding + Qdrant upsert.
        await self._vector.aadd(msgs)  # type: ignore[arg-type]
        # Index in BM25 for the lexical channel.
        for msg in msgs:
            self._bm25.add(self.session_id, msg.id, msg.content or "")

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        bm25_hits = self._bm25.search(self.session_id, query, k=k)
        bm25_ids = [msg_id for msg_id, _ in bm25_hits]

        vector_results = await self._vector.asearch(query, k=k)
        vector_ids = [r.message.id for r in vector_results]

        fused_ids = rrf([bm25_ids, vector_ids], k=self._rrf_k)[:k]

        all_msgs = await self._store.get_all(self.session_id)
        msg_by_id = {m.id: m for m in all_msgs}

        return [
            RetrievalResult(message=msg_by_id[did], score=1.0 / (self._rrf_k + rank + 1))
            for rank, did in enumerate(fused_ids)
            if did in msg_by_id
        ]

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        k = limit or self._top_k
        if query is None:
            return await self._store.recent(self.session_id, k)
        results = await self.asearch(query, k=k)
        return sorted([r.message for r in results], key=lambda m: m.timestamp)

    async def aclear(self) -> None:
        await self._vector.aclear()
        self._bm25.clear(self.session_id)
