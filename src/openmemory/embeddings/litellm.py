"""LiteLLM embedder ([litellm] extra).

One configuration surface for 100+ providers (OpenAI, Azure, Bedrock, Vertex, Cohere,
Ollama, ...). The model string follows LiteLLM conventions, e.g. ``"text-embedding-3-small"``
or ``"ollama/nomic-embed-text"``.
"""

from __future__ import annotations

from ..config import Config
from .base import Embedder


class LiteLLMEmbedder(Embedder):
    def __init__(self, config: Config) -> None:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LiteLLM embedder requires the 'litellm' extra. "
                "Install with: pip install 'open-memory[litellm]'"
            ) from exc
        self._litellm = litellm
        self._model = config.embedding_model
        self._dim = config.embedding_dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._litellm.aembedding(model=self._model, input=texts)
        vectors = [item["embedding"] for item in resp["data"]]
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
