"""WindowMemory — keep only the most recent turns.

Caps context by message count (``n``) and/or a token budget. A leading ``system``
message is always preserved so persona/instructions survive truncation.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult
from ..core.tokens import count_message_tokens
from ..storage.session_store import SessionStore


class WindowMemory(BaseMemory):
    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        *,
        n: int | None = 20,
        token_budget: int | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._n = n
        self._token_budget = token_budget
        self._model = model

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
        if not messages:
            return []

        # Preserve a leading system message regardless of windowing.
        system: Message | None = None
        if messages[0].role == "system":
            system, messages = messages[0], messages[1:]

        n = limit if limit is not None else self._n
        if n is not None:
            messages = messages[-n:]

        budget = token_budget if token_budget is not None else self._token_budget
        if budget is not None:
            messages = self._fit_budget(messages, budget, system)

        return ([system] + messages) if system else messages

    def _fit_budget(
        self, messages: list[Message], budget: int, system: Message | None
    ) -> list[Message]:
        used = count_message_tokens(system, self._model) if system else 0
        kept: list[Message] = []
        for msg in reversed(messages):  # newest first
            cost = count_message_tokens(msg, self._model)
            if used + cost > budget:
                break
            used += cost
            kept.append(msg)
        kept.reverse()
        return kept

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return []

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
