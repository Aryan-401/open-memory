"""Shared test fixtures and offline fakes.

The vector/hybrid/hierarchical strategies need an embedder (and the hierarchical one an
LLM). To keep the suite hermetic — no network, no API keys, no Docker — we provide a
deterministic bag-of-words embedder (so cosine similarity tracks word overlap) and a
trivial summarizing LLM.
"""

from __future__ import annotations

import hashlib
import math
import re

import pytest

from openmemory.embeddings.base import Embedder
from openmemory.llm.base import LLM
from openmemory.storage.qdrant_store import QdrantVectorStore
from openmemory.storage.session_store import InMemorySessionStore

_DIM = 64
_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words embedding: word overlap -> high cosine similarity."""

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @property
    def dim(self) -> int:
        return _DIM


class FakeLLM(LLM):
    """Echoes the turns it is asked to summarize so summaries are assertable."""

    def __init__(self) -> None:
        self.calls = 0

    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        self.calls += 1
        user = messages[-1]["content"]
        return f"SUMMARY({self.calls}): {user[:200]}"


@pytest.fixture
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def vectors() -> QdrantVectorStore:
    vs = QdrantVectorStore()  # in-memory Qdrant
    yield vs
    await vs.close()
