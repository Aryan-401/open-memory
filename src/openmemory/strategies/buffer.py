"""BufferMemory — the naive "full list of dicts" strategy.

Stores every message and replays the entire history. Simple and lossless; appropriate
for short conversations or when the caller manages truncation elsewhere.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult
from ..storage.session_store import SessionStore


class BufferMemory(BaseMemory):
    def __init__(self, session_id: str, store: SessionStore) -> None:
        self.session_id = session_id
        self._store = store

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        await self._store.append(self.session_id, self._coerce(messages))

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        messages = await self._store.get_all(self.session_id)
        if limit is not None:
            messages = messages[-limit:]
        return messages

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        # No embeddings in this strategy.
        return []

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
