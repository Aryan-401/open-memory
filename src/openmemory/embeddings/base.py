"""Embedder interface and factory.

Backends lazily import their provider library so the core install stays lightweight;
selecting a backend whose extra isn't installed raises a clear, actionable error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config


class Embedder(ABC):
    """Turns text into vectors for semantic retrieval."""

    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    async def aembed_one(self, text: str) -> list[float]:
        return (await self.aembed([text]))[0]

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""


def build_embedder(config: Config) -> Embedder:
    """Construct the embedder selected in ``config.embedder_provider``."""
    provider = config.embedder_provider
    if provider == "openai":
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder(config)
    if provider == "litellm":
        from .litellm import LiteLLMEmbedder

        return LiteLLMEmbedder(config)
    if provider == "local":
        from .local import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(config)
    raise ValueError(f"Unknown embedder provider: {provider!r}")
