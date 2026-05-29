"""Session-scoped storage of full message records (recall storage).

The ``SessionStore`` persists complete :class:`Message` objects (all internal fields
intact) keyed by ``session_id``. This is distinct from the vector store, which holds
embeddings for semantic retrieval. The in-memory implementation is the default; the
SQLite implementation (``sqlite_store``) adds durability across restarts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from ..core.models import Message


class SessionStore(ABC):
    """Append-and-read store for messages, partitioned by session."""

    @abstractmethod
    async def append(self, session_id: str, messages: list[Message]) -> None: ...

    @abstractmethod
    async def get_all(self, session_id: str) -> list[Message]:
        """Return all messages for a session in insertion order."""

    @abstractmethod
    async def recent(self, session_id: str, n: int) -> list[Message]:
        """Return the most recent ``n`` messages in chronological order."""

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...

    @abstractmethod
    async def sessions(self) -> list[str]:
        """List known session ids."""


class InMemorySessionStore(SessionStore):
    """Process-local store. Fast and dependency-free; not durable."""

    def __init__(self) -> None:
        self._data: dict[str, list[Message]] = defaultdict(list)

    async def append(self, session_id: str, messages: list[Message]) -> None:
        self._data[session_id].extend(messages)

    async def get_all(self, session_id: str) -> list[Message]:
        return list(self._data.get(session_id, []))

    async def recent(self, session_id: str, n: int) -> list[Message]:
        return list(self._data.get(session_id, []))[-n:]

    async def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    async def sessions(self) -> list[str]:
        return list(self._data.keys())
