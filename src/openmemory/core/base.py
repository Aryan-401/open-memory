"""The strategy contract every memory backend implements.

``BaseMemory`` is async-first: subclasses implement the ``a*`` coroutines. Synchronous
wrappers are provided here for convenience and delegate through
:func:`openmemory._sync.run_sync`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .._sync import run_sync
from .models import Message, RetrievalResult, coerce_messages, to_openai_messages


class BaseMemory(ABC):
    """Base class for all context-management strategies, scoped to one session."""

    session_id: str

    def _coerce(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> list[Message]:
        return coerce_messages(messages, session_id=self.session_id)

    # --- Async interface (implemented by subclasses) ---

    @abstractmethod
    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        """Add one or more messages to memory."""

    @abstractmethod
    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        """Assemble the context to feed the model for the next turn."""

    @abstractmethod
    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Semantic search over stored messages. Strategies without embeddings may
        return an empty list."""

    @abstractmethod
    async def aclear(self) -> None:
        """Remove all messages for this session."""

    async def aget_openai_context(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Convenience: ``aget_context`` serialized to OpenAI chat dicts."""
        return to_openai_messages(await self.aget_context(**kwargs))

    # --- Sync wrappers ---

    def add(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        return run_sync(self.aadd(messages))

    def get_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        return run_sync(
            self.aget_context(query=query, limit=limit, token_budget=token_budget)
        )

    def get_openai_context(self, **kwargs: Any) -> list[dict[str, Any]]:
        return run_sync(self.aget_openai_context(**kwargs))

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return run_sync(self.asearch(query, k))

    def clear(self) -> None:
        return run_sync(self.aclear())
