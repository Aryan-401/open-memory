"""Tests for GraphMemory and NetworkxGraphStore."""

from __future__ import annotations

import pytest

from openmemory.llm.base import LLM
from openmemory.strategies.graph import GraphMemory, NetworkxGraphStore


class TripletLLM(LLM):
    """Returns a fixed list of triplets."""

    def __init__(self, triplets: list[list[str]]) -> None:
        self._triplets = triplets

    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        import json
        return json.dumps(self._triplets)


class EmptyTripletLLM(LLM):
    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        return "[]"


class BadJsonLLM(LLM):
    async def achat(self, messages, *, temperature: float = 0.2) -> str:
        return "not json"


def _mem(store, llm, hops: int = 1) -> tuple[GraphMemory, NetworkxGraphStore]:
    g = NetworkxGraphStore()
    return GraphMemory("g-test", store, llm, g, hops=hops), g


async def test_triplets_added_to_graph(store):
    """Triplets extracted by the LLM are added to the networkx graph."""
    llm = TripletLLM([["Alice", "works_with", "Bob"]])
    mem, g = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "Alice works with Bob."})
    assert g.edge_count("g-test") == 1
    assert g.node_count("g-test") == 2


async def test_context_no_query_returns_all_triplets(store):
    """aget_context() with no query returns the full graph as a system message."""
    llm = TripletLLM([["Alice", "works_with", "Bob"], ["project", "deadline_is", "Friday"]])
    mem, _ = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "content"})
    ctx = await mem.aget_context()
    assert len(ctx) == 1
    assert ctx[0].role == "system"
    assert "Alice" in ctx[0].content
    assert "project" in ctx[0].content


async def test_context_with_query_traverses_neighbourhood(store):
    """aget_context(query=...) returns only triplets near entities in the query."""
    llm = TripletLLM([
        ["Alice", "works_with", "Bob"],
        ["Carol", "manages", "Dave"],
    ])
    mem, _ = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "org structure"})
    # Query contains "Alice" — should return Alice-related triplets
    ctx = await mem.aget_context(query="Who does Alice work with?")
    assert len(ctx) == 1
    assert "Alice" in ctx[0].content
    # Carol's triplet may or may not appear depending on graph distance


async def test_context_empty_when_no_graph(store):
    """aget_context returns [] when no triplets have been extracted."""
    llm = EmptyTripletLLM()
    mem, _ = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "nothing here"})
    ctx = await mem.aget_context()
    assert ctx == []


async def test_bad_json_from_llm_gracefully_ignored(store):
    """Malformed JSON from the triplet LLM doesn't raise."""
    llm = BadJsonLLM()
    mem, g = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "something"})
    assert g.edge_count("g-test") == 0
    ctx = await mem.aget_context()
    assert ctx == []


async def test_asearch_returns_empty(store):
    """GraphMemory has no vector index; asearch always returns []."""
    llm = TripletLLM([["A", "rel", "B"]])
    mem, _ = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "A relates to B."})
    results = await mem.asearch("A")
    assert results == []


async def test_clear_resets_graph(store):
    """aclear removes the session's graph and session store entries."""
    llm = TripletLLM([["X", "knows", "Y"]])
    mem, g = _mem(store, llm)
    await mem.aadd({"role": "user", "content": "X knows Y."})
    assert g.edge_count("g-test") == 1

    await mem.aclear()
    assert g.edge_count("g-test") == 0
    ctx = await mem.aget_context()
    assert ctx == []


async def test_first_write_wins_for_duplicate_edges(store):
    """Adding the same subject-object pair twice keeps the first predicate."""
    g = NetworkxGraphStore()
    await g.add_triplets("s", [("A", "first_rel", "B")])
    await g.add_triplets("s", [("A", "second_rel", "B")])
    triplets = g.all_triplets("s")
    assert len(triplets) == 1
    assert triplets[0][1] == "first_rel"


async def test_sessions_isolated(store):
    """Two GraphMemory instances with different session_ids don't share graph state."""
    g_store = NetworkxGraphStore()
    llm1 = TripletLLM([["Alice", "knows", "Bob"]])
    llm2 = TripletLLM([["Carol", "manages", "Dave"]])
    m1 = GraphMemory("sess-1", store, llm1, g_store)
    m2 = GraphMemory("sess-2", store, llm2, g_store)
    await m1.aadd({"role": "user", "content": "Alice knows Bob."})
    await m2.aadd({"role": "user", "content": "Carol manages Dave."})

    t1 = g_store.all_triplets("sess-1")
    t2 = g_store.all_triplets("sess-2")
    assert any("Alice" in t[0] for t in t1)
    assert not any("Alice" in t[0] for t in t2)


async def test_hops_expands_neighbourhood(store):
    """With hops=2, the traversal reaches two steps from the matched entity."""
    g = NetworkxGraphStore()
    await g.add_triplets("s", [
        ("Alice", "works_with", "Bob"),
        ("Bob", "manages", "Carol"),
        ("Dave", "unrelated", "Eve"),
    ])
    triplets = await g.neighborhood("s", ["Alice"], hops=2)
    subjects = {t[0] for t in triplets} | {t[2] for t in triplets}
    assert "Carol" in subjects  # 2 hops from Alice
    assert "Eve" not in subjects  # completely disconnected
