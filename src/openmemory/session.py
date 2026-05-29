"""A :class:`Session` binds a strategy instance to a ``session_id`` and carries metadata.

It forwards the full memory API (async + sync) to the underlying strategy, so callers
work with one object per conversation regardless of which strategy backs it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .core.base import BaseMemory
from .core.models import Message, RetrievalResult


class Session:
    def __init__(
        self,
        session_id: str,
        memory: BaseMemory,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.memory = memory
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)

    # --- Async API (delegated) ---
    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        await self.memory.aadd(messages)

    async def aget_context(self, **kwargs: Any) -> list[Message]:
        return await self.memory.aget_context(**kwargs)

    async def aget_openai_context(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.memory.aget_openai_context(**kwargs)

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return await self.memory.asearch(query, k)

    async def aclear(self) -> None:
        await self.memory.aclear()

    # --- Sync API (delegated) ---
    def add(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        self.memory.add(messages)

    def get_context(self, **kwargs: Any) -> list[Message]:
        return self.memory.get_context(**kwargs)

    def get_openai_context(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.memory.get_openai_context(**kwargs)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return self.memory.search(query, k)

    def clear(self) -> None:
        self.memory.clear()

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id!r}, "
            f"strategy={type(self.memory).__name__})"
        )
