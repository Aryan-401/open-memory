"""OpenAI-compatible embedder (core).

Works against the official OpenAI API or any compatible endpoint (vLLM, Ollama,
LiteLLM proxy, ...) via ``config.openai_base_url``.
"""

from __future__ import annotations

from ..config import Config
from .base import Embedder

# Known dimensions for common OpenAI models; otherwise inferred from the first response.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    def __init__(self, config: Config) -> None:
        from openai import AsyncOpenAI

        self._model = config.embedding_model
        self._client = AsyncOpenAI(
            api_key=config.openai_api_key, base_url=config.openai_base_url
        )
        self._dim = config.embedding_dim or _KNOWN_DIMS.get(self._model)

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in resp.data]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError(
                "Embedding dimension unknown until the first embed call; "
                "set config.embedding_dim to declare it up front."
            )
        return self._dim
