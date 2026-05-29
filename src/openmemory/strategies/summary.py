"""SummaryMemory — SCAFFOLD (not yet implemented).

Planned behavior: maintain a rolling summary of the whole conversation plus a small
recent buffer (LangChain's ConversationSummaryBufferMemory pattern). On each add, fold
turns beyond the buffer into the summary. ``aget_context`` returns
``[summary-as-system, *recent-buffer]``.

The summarization machinery already exists in :class:`openmemory.llm.summarizer.Summarizer`
and is exercised by :class:`~openmemory.strategies.hierarchical.HierarchicalMemory`; this
standalone strategy will wrap it without the archival/retrieval tiers.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult

_NOT_READY = (
    "SummaryMemory is scaffolded but not implemented yet. "
    "Use HierarchicalMemory for summary-backed context today."
)


class SummaryMemory(BaseMemory):
    def __init__(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        self.session_id = session_id

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        raise NotImplementedError(_NOT_READY)

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        raise NotImplementedError(_NOT_READY)

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError(_NOT_READY)

    async def aclear(self) -> None:
        raise NotImplementedError(_NOT_READY)
