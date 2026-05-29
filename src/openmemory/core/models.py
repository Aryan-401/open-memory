"""Core data models shared by every strategy.

A :class:`Message` carries the OpenAI-compliant ``role``/``content`` (plus optional
``name``/``tool_calls``) *and* rich internal fields that strategies query and filter on
(``id``, ``session_id``, ``timestamp``, ``tags``, ``importance``, ``metadata``,
``embedding``). The internal fields are stripped on serialization so what reaches the
model is always a clean ``{"role", "content"}`` dict.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
EmbedMode = Literal["per_message", "paired"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class Message(BaseModel):
    """A single conversational turn with both wire fields and internal metadata."""

    # --- OpenAI-compliant wire fields ---
    role: Role
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    # --- Internal, queryable fields (filtered out on serialization) ---
    id: str = Field(default_factory=_new_id)
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None, repr=False)

    def to_openai(self) -> dict[str, Any]:
        """Return the OpenAI chat-completions message dict (wire fields only)."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            msg["name"] = self.name
        if self.tool_calls is not None:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def embedding_text(self) -> str:
        """Text used to compute this message's embedding."""
        return self.content


def embed_texts(msgs: list[Message], mode: EmbedMode = "per_message") -> list[str]:
    """Return one embedding string per message, respecting the chosen embed mode.

    ``"per_message"`` — each message embeds its own content (default).

    ``"paired"`` — consecutive user+assistant pairs share a turn-level embedding
    for the user message (``"User: ...\\nAssistant: ..."``), giving the user vector
    richer semantic surface area at search time. The assistant message still gets its
    own vector so it remains independently retrievable. Unpaired messages (system,
    tool, a trailing user message with no reply yet) embed normally.
    """
    if mode == "per_message":
        return [m.embedding_text() for m in msgs]

    texts: list[str] = []
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if (
            msg.role == "user"
            and i + 1 < len(msgs)
            and msgs[i + 1].role == "assistant"
        ):
            texts.append(f"User: {msg.content}\nAssistant: {msgs[i + 1].content}")
            texts.append(msgs[i + 1].embedding_text())
            i += 2
        else:
            texts.append(msg.embedding_text())
            i += 1
    return texts


class RetrievalResult(BaseModel):
    """A message returned from semantic search, with its similarity score."""

    message: Message
    score: float


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Serialize messages to the OpenAI chat format, dropping all internal fields."""
    return [m.to_openai() for m in messages]


def coerce_messages(
    items: Message | dict[str, Any] | list[Message | dict[str, Any]],
    *,
    session_id: str | None = None,
) -> list[Message]:
    """Normalize a Message, a raw ``{role, content}`` dict, or a list thereof to Messages.

    Stamps ``session_id`` on any message that does not already carry one.
    """
    if isinstance(items, (Message, dict)):
        items = [items]

    out: list[Message] = []
    for item in items:
        msg = item if isinstance(item, Message) else Message(**item)
        if msg.session_id is None:
            msg.session_id = session_id
        out.append(msg)
    return out
