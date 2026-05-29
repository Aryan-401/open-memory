"""Local sentence-transformers embedder ([local-llm] extra).

Runs fully offline. The model is synchronous, so encoding is dispatched to a worker
thread via ``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio

from ..config import Config
from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, config: Config) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Local embedder requires the 'local-llm' extra. "
                "Install with: pip install 'open-memory[local-llm]'"
            ) from exc
        self._model = SentenceTransformer(config.local_embedding_model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        return self._dim
