"""SummaryMemory — rolling summary + verbatim recent buffer.

Two tiers, no vector store:
  * **recent buffer** — the last ``buffer_size`` non-system messages kept verbatim;
  * **rolling summary** — everything older, compressed by the LLM into one block.

On each add, any messages that have aged out of the buffer are folded into the summary
via :class:`~openmemory.llm.summarizer.Summarizer`. ``aget_context`` returns
``[system] + [summary-as-system] + [buffer]``.

This is the LangChain ConversationSummaryBufferMemory pattern — a lighter alternative to
:class:`~openmemory.strategies.hierarchical.HierarchicalMemory` when archival retrieval
is not needed.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult
from ..llm.summarizer import Summarizer
from ..storage.session_store import SessionStore


class SummaryMemory(BaseMemory):
    """Rolling-summary + recent-buffer memory. No vector store required.

    Parameters
    ----------
    buffer_size:
        Number of most-recent *non-system* messages to keep verbatim. Once the body
        exceeds this size, the oldest overflow is folded into the rolling summary.
    """

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        summarizer: Summarizer,
        *,
        buffer_size: int = 6,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._summarizer = summarizer
        self._buffer_size = buffer_size
        self._summary: str = ""
        self._summarized_count: int = 0  # messages folded into summary so far

    def _split_system(
        self, messages: list[Message]
    ) -> tuple[Message | None, list[Message]]:
        if messages and messages[0].role == "system":
            return messages[0], messages[1:]
        return None, messages

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        msgs = self._coerce(messages)
        if not msgs:
            return
        await self._store.append(self.session_id, msgs)
        await self._maybe_compress()

    async def _maybe_compress(self) -> None:
        all_msgs = await self._store.get_all(self.session_id)
        _, body = self._split_system(all_msgs)
        cutoff = len(body) - self._buffer_size
        if cutoff > self._summarized_count:
            to_fold = body[self._summarized_count:cutoff]
            self._summary = await self._summarizer.asummarize(to_fold, self._summary)
            self._summarized_count = cutoff

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        all_msgs = await self._store.get_all(self.session_id)
        system, body = self._split_system(all_msgs)
        buffer = body[self._summarized_count:]  # verbatim tail

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
        out.extend(buffer)
        return out

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        # SummaryMemory has no vector index; semantic search is not supported.
        return []

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
        self._summary = ""
        self._summarized_count = 0
