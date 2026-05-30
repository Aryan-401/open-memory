"""Tests for FactExtractionMemory."""

from __future__ import annotations

import pytest

from openmemory.llm.base import LLM
from openmemory.storage.qdrant_store import QdrantVectorStore
from openmemory.strategies.facts import FactExtractionMemory


class FactLLM(LLM):
    """Always returns a fixed set of fact strings."""

    def __init__(self, facts: list[str]) -> None:
        self._facts = facts

    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        import json
        return json.dumps(self._facts)


class EmptyFactLLM(LLM):
    """Returns no facts."""

    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        return "[]"


class BadJsonLLM(LLM):
    """Returns malformed JSON."""

    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        return "not valid json at all"


@pytest.fixture
async def fact_vectors() -> QdrantVectorStore:
    vs = QdrantVectorStore(collection="test_facts")
    yield vs
    await vs.close()


def _mem(store, llm, embedder, fact_vectors, **kwargs) -> FactExtractionMemory:
    return FactExtractionMemory(
        "f-test", store, llm, embedder, fact_vectors, **kwargs
    )


async def test_facts_extracted_and_stored(store, embedder, fact_vectors):
    """Facts are extracted from turns and stored in Qdrant."""
    llm = FactLLM(["user is a developer", "user likes Python"])
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "I'm a developer who loves Python."})
    assert len(mem._fact_texts) == 2


async def test_context_without_query_returns_all_facts(store, embedder, fact_vectors):
    """aget_context() with no query returns all known facts as one system message."""
    llm = FactLLM(["fact one", "fact two"])
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "some content"})
    ctx = await mem.aget_context()
    assert len(ctx) == 1
    assert ctx[0].role == "system"
    assert "fact one" in ctx[0].content
    assert "fact two" in ctx[0].content


async def test_context_with_query_does_semantic_search(store, embedder, fact_vectors):
    """aget_context(query=...) does semantic search over facts."""
    llm = FactLLM(["user likes Python"])
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "I love Python"})
    ctx = await mem.aget_context(query="programming language preferences")
    assert len(ctx) == 1
    assert "Python" in ctx[0].content


async def test_empty_context_when_no_facts(store, embedder, fact_vectors):
    """aget_context returns [] when no facts have been extracted yet."""
    llm = EmptyFactLLM()
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "nothing extractable"})
    ctx = await mem.aget_context()
    assert ctx == []


async def test_deduplication_prevents_duplicate_facts(store, embedder, fact_vectors):
    """A near-identical fact is not stored twice (cosine similarity dedup)."""
    llm = FactLLM(["user likes Python"])
    mem = _mem(store, llm, embedder, fact_vectors, dedup_threshold=0.7)
    await mem.aadd({"role": "user", "content": "I love Python"})
    # Same fact — should be deduplicated.
    await mem.aadd({"role": "user", "content": "I love Python"})
    # Due to FakeEmbedder determinism and exact same text → exact same vector → dedup.
    assert len(mem._fact_texts) == 1


async def test_bad_llm_response_gracefully_ignored(store, embedder, fact_vectors):
    """Malformed JSON from the LLM doesn't raise — extraction returns []."""
    llm = BadJsonLLM()
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "something"})
    assert mem._fact_texts == []
    ctx = await mem.aget_context()
    assert ctx == []


async def test_asearch_returns_results(store, embedder, fact_vectors):
    """asearch embeds the query and returns matching facts from Qdrant."""
    llm = FactLLM(["user prefers dark mode"])
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "I like dark mode."})
    results = await mem.asearch("dark mode preference", k=5)
    assert len(results) > 0
    assert "dark mode" in results[0].message.content


async def test_clear_resets_state(store, embedder, fact_vectors):
    """aclear empties the store, Qdrant, and the in-process fact list."""
    llm = FactLLM(["some fact"])
    mem = _mem(store, llm, embedder, fact_vectors)
    await mem.aadd({"role": "user", "content": "content"})
    assert len(mem._fact_texts) == 1

    await mem.aclear()
    assert mem._fact_texts == []
    ctx = await mem.aget_context()
    assert ctx == []


async def test_sessions_isolated(store, embedder):
    """Two sessions use separate Qdrant stores and don't share facts."""
    vs1 = QdrantVectorStore(collection="test_facts_sess1")
    vs2 = QdrantVectorStore(collection="test_facts_sess2")
    try:
        m1 = FactExtractionMemory("s1", store, FactLLM(["fact A"]), embedder, vs1)
        m2 = FactExtractionMemory("s2", store, FactLLM(["fact B"]), embedder, vs2)
        await m1.aadd({"role": "user", "content": "A"})
        await m2.aadd({"role": "user", "content": "B"})
        assert "fact A" in m1._fact_texts
        assert "fact B" in m2._fact_texts
        assert "fact A" not in m2._fact_texts
    finally:
        await vs1.close()
        await vs2.close()
