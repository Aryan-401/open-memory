"""GraphMemory — SCAFFOLD (not yet implemented).

Planned behavior: build a knowledge graph of entities and their relationships extracted
from the conversation (Zep/Graphiti / mem0-graph style), and retrieve context by
traversing the subgraph around entities mentioned in the query. This captures relational
structure ("who works with whom", "which task blocks which") that flat vector retrieval
misses.

A ``GraphStore`` protocol is sketched below so a ``networkx`` in-process backend can land
first, with a Neo4j/Graphiti backend slotting in behind the same interface later.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult

_NOT_READY = (
    "GraphMemory is scaffolded but not implemented yet. "
    "A networkx backend is planned, followed by Neo4j/Graphiti."
)


@runtime_checkable
class GraphStore(Protocol):
    """Backend contract for graph-based memory (networkx, Neo4j, Graphiti, ...)."""

    async def add_triplets(
        self, session_id: str, triplets: list[tuple[str, str, str]]
    ) -> None: ...

    async def neighborhood(
        self, session_id: str, entities: list[str], hops: int = 1
    ) -> list[tuple[str, str, str]]: ...

    async def clear(self, session_id: str) -> None: ...


class GraphMemory(BaseMemory):
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
