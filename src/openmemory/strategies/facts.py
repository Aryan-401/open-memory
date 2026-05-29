"""FactExtractionMemory — SCAFFOLD (not yet implemented).

Planned behavior (mem0-style): instead of storing raw turns, run each new turn through
an LLM extractor that distills durable *facts* / preferences ("user is vegetarian",
"deadline is Friday"). Facts are deduplicated/updated (ADD/UPDATE/DELETE reconciliation
against semantically similar existing facts) and stored in the vector index.
``aget_context`` retrieves the facts most relevant to the current query.

Integration points already available: the ``LLM`` interface (``openmemory.llm.base``)
for extraction, plus ``Embedder`` and ``QdrantVectorStore`` for fact storage/retrieval.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult

_NOT_READY = (
    "FactExtractionMemory is scaffolded but not implemented yet. "
    "Track progress in the project roadmap."
)


class FactExtractionMemory(BaseMemory):
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
