"""LLM interface and factory (used for summarization and, later, fact extraction)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import Config


class LLM(ABC):
    """Minimal async chat interface."""

    @abstractmethod
    async def achat(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.2
    ) -> str:
        """Return the assistant's text reply for the given chat messages."""


def build_llm(config: Config) -> LLM:
    """Construct the LLM selected in ``config.llm_provider``."""
    provider = config.llm_provider
    if provider == "openai":
        from .openai import OpenAILLM

        return OpenAILLM(config)
    if provider == "litellm":
        from .litellm import LiteLLMLLM

        return LiteLLMLLM(config)
    if provider == "local":
        from .local import LocalHFLLM

        return LocalHFLLM(config)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
