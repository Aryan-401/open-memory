"""Token counting for budget-aware windowing.

Uses ``tiktoken`` when a matching encoding is available, falling back to a cheap
character-based heuristic so the library never hard-fails on an unknown model.
"""

from __future__ import annotations

from functools import lru_cache

from .models import Message

# Rough per-message overhead in the OpenAI chat format (role + delimiters).
_PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=8)
def _encoding(model: str):
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is a core dep
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Approximate token count for a string."""
    enc = _encoding(model)
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def count_message_tokens(message: Message, model: str = "gpt-4o-mini") -> int:
    """Approximate token count for a single chat message, including format overhead."""
    return count_text_tokens(message.content, model) + _PER_MESSAGE_OVERHEAD


def count_messages_tokens(messages: list[Message], model: str = "gpt-4o-mini") -> int:
    """Approximate token count for a list of messages."""
    return sum(count_message_tokens(m, model) for m in messages)
