"""FullHybridMemory — three-way hybrid: recency window + BM25 + dense vector via RRF.

Combines all three retrieval signals that the other hybrid strategies each cover partially:

- :class:`~openmemory.strategies.hybrid.HybridMemory` — recency window ∪ semantic ANN
  (no keyword precision)
- :class:`~openmemory.strategies.sparse_hybrid.SparseHybridMemory` — BM25 ∪ semantic ANN
  (no recency guarantee)
- **FullHybridMemory** — recency window ∪ BM25 ∪ semantic ANN (all three, three-way RRF)

At query time, each signal independently ranks the session's messages; Reciprocal Rank
Fusion aggregates the three rankings. Messages that appear highly in multiple signals are
promoted; messages strong in only one still surface. The fused list is sorted
chronologically before being returned (rank-to-select, time-to-present).

Add path delegates to :class:`~openmemory.strategies.sparse_hybrid.SparseHybridMemory`
which writes to the session store, Qdrant, and BM25 index in one shot — the recency
window reads from the same session store at query time, so there is no separate write.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import EmbedMode, Message, RetrievalResult
from ..embeddings.base import Embedder
from ..storage.bm25_store import BM25Store, rrf
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.session_store import SessionStore
from .sparse_hybrid import SparseHybridMemory
from .window import WindowMemory


class FullHybridMemory(BaseMemory):
    """Three-way hybrid retrieval: recency window ∪ BM25 ∪ dense vector, fused via RRF.

    Parameters
    ----------
    window_size:
        Number of recent messages included as the recency signal in RRF.
    top_k:
        Number of candidates fetched from each retrieval channel before fusion.
        The final output is also capped at ``top_k``.
    embed_mode:
        ``"per_message"`` or ``"paired"`` — forwarded to the vector channel.
    rrf_k:
        RRF rank-smoothing constant (default 60).
    model:
        Tiktoken model used for optional token-budget enforcement on the recency window.
    token_budget:
        Optional token cap on the recency window. ``None`` = no cap.
    """

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        bm25_store: BM25Store,
        *,
        window_size: int = 10,
        top_k: int = 5,
        embed_mode: EmbedMode = "per_message",
        model: str = "gpt-4o-mini",
        token_budget: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._sparse = SparseHybridMemory(
            session_id,
            store,
            embedder,
            vector_store,
            bm25_store,
            top_k=top_k,
            embed_mode=embed_mode,
            rrf_k=rrf_k,
        )
        self._window = WindowMemory(
            session_id, store, n=window_size, token_budget=token_budget, model=model
        )
        self._top_k = top_k
        self._rrf_k = rrf_k

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        # SparseHybridMemory handles session store + Qdrant + BM25 in one pass.
        await self._sparse.aadd(messages)

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return await self._sparse.asearch(query, k=k)

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        k = limit or self._top_k

        # Use the window's own configured n; don't override with k (retrieval candidate count).
        window_msgs = await self._window.aget_context(token_budget=token_budget)
        window_ids = [m.id for m in window_msgs]

        if query is None:
            return window_msgs

        bm25_hits = self._sparse._bm25.search(self.session_id, query, k=k)
        bm25_ids = [msg_id for msg_id, _ in bm25_hits]

        vector_results = await self._sparse._vector.asearch(query, k=k)
        vector_ids = [r.message.id for r in vector_results]

        fused_ids = rrf([window_ids, bm25_ids, vector_ids], k=self._rrf_k)[:k]

        all_msgs = await self._store.get_all(self.session_id)
        msg_by_id = {m.id: m for m in all_msgs}

        selected = [msg_by_id[did] for did in fused_ids if did in msg_by_id]
        return sorted(selected, key=lambda m: m.timestamp)

    async def aclear(self) -> None:
        await self._sparse.aclear()
