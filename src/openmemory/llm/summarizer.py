"""Conversation summarization building block.

Used by :class:`~openmemory.strategies.hierarchical.HierarchicalMemory` to compress
evicted turns into a rolling summary. Folds new turns into any existing summary so the
summary stays bounded regardless of conversation length.
"""

from __future__ import annotations

from ..core.models import Message, to_openai_messages
from .base import LLM

_SYSTEM = (
    "You compress conversation history into a concise running summary. "
    "Preserve durable facts, decisions, user preferences, names, and open tasks. "
    "Drop chit-chat. Return only the updated summary."
)


class Summarizer:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def asummarize(
        self, messages: list[Message], existing_summary: str | None = None
    ) -> str:
        """Fold ``messages`` into ``existing_summary`` and return the new summary."""
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_openai_messages(messages)
        )
        user_parts = []
        if existing_summary:
            user_parts.append(f"Current summary:\n{existing_summary}")
        user_parts.append(f"New turns to fold in:\n{transcript}")
        chat = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        return await self._llm.achat(chat)
