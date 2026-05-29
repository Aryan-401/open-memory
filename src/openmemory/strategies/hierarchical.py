"""HierarchicalMemory — MemGPT/Letta-style paged memory.

Three tiers:
  * **working context** — a token-bounded tail of recent turns kept verbatim;
  * **recall storage** — the full ordered history (session store);
  * **archival storage** — embeddings of every turn (Qdrant) for query-time retrieval.

When the working context overflows its token budget, the turns that fall out of it are
folded into a recursive **rolling summary** (via :class:`Summarizer`). They remain
retrievable from archival, so nothing is lost. ``aget_context`` assembles:
``system + summary + archival-retrieved(query) + working-tail`` within ``token_budget``.

The rolling summary is held in process on the strategy instance (v1); restart-durable
summaries are a future enhancement.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import EmbedMode, Message, RetrievalResult, embed_texts
from ..core.tokens import count_message_tokens
from ..embeddings.base import Embedder
from ..llm.summarizer import Summarizer
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.session_store import SessionStore


class HierarchicalMemory(BaseMemory):
    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        summarizer: Summarizer,
        *,
        working_context_tokens: int = 2000,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        embed_mode: EmbedMode = "per_message",
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._embedder = embedder
        self._vectors = vector_store
        self._summarizer = summarizer
        self._working_tokens = working_context_tokens
        self._top_k = top_k
        self._model = model
        self._embed_mode = embed_mode

        self._summary: str = ""
        self._summarized_count: int = 0  # number of leading non-system msgs folded in

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        msgs = self._coerce(messages)
        if not msgs:
            return
        vectors = await self._embedder.aembed(embed_texts(msgs, self._embed_mode))
        await self._store.append(self.session_id, msgs)
        await self._vectors.upsert(msgs, vectors)
        await self._maybe_evict()

    def _split_system(self, messages: list[Message]) -> tuple[Message | None, list[Message]]:
        if messages and messages[0].role == "system":
            return messages[0], messages[1:]
        return None, messages

    def _working_start(self, body: list[Message]) -> int:
        """Index into ``body`` where the working tail begins (fits the token budget)."""
        used = 0
        start = len(body)
        for i in range(len(body) - 1, -1, -1):
            used += count_message_tokens(body[i], self._model)
            if used > self._working_tokens:
                break
            start = i
        return start

    async def _maybe_evict(self) -> None:
        """Fold any turns that have aged out of the working context into the summary."""
        all_msgs = await self._store.get_all(self.session_id)
        _, body = self._split_system(all_msgs)
        start = self._working_start(body)
        if start > self._summarized_count:
            to_fold = body[self._summarized_count : start]
            self._summary = await self._summarizer.asummarize(to_fold, self._summary)
            self._summarized_count = start

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
        all_msgs = await self._store.get_all(self.session_id)
        system, body = self._split_system(all_msgs)
        start = self._working_start(body)
        working = body[start:]
        working_ids = {m.id for m in working}

        out: list[Message] = []
        if system:
            out.append(system)
        if self._summary:
            out.append(
                Message(
                    role="system",
                    content=f"Summary of earlier conversation:\n{self._summary}",
                    session_id=self.session_id,
                    metadata={"openmemory_kind": "summary"},
                )
            )

        if query is not None:
            results = await self.asearch(query, k=limit or self._top_k)
            retrieved = [
                r.message for r in results if r.message.id not in working_ids
            ]
            out.extend(sorted(retrieved, key=lambda m: m.timestamp))

        out.extend(working)
        return out

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
        await self._vectors.clear(self.session_id)
        self._summary = ""
        self._summarized_count = 0
