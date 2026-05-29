from __future__ import annotations

from openmemory import OpenMemory
from openmemory.strategies.hybrid import HybridMemory
from openmemory.strategies.vector import VectorMemory


async def test_vector_retrieves_relevant_over_distractor(store, embedder, vectors):
    mem = VectorMemory("s1", store, embedder, vectors, top_k=1)
    await mem.aadd({"role": "user", "content": "I love hiking mountains and trails"})
    await mem.aadd({"role": "user", "content": "my favorite food is spicy ramen noodles"})

    results = await mem.asearch("tell me about hiking trails", k=1)
    assert len(results) == 1
    assert "hiking" in results[0].message.content


async def test_vector_sessions_isolated(store, embedder, vectors):
    a = VectorMemory("a", store, embedder, vectors)
    b = VectorMemory("b", store, embedder, vectors)
    await a.aadd({"role": "user", "content": "alpha hiking"})
    await b.aadd({"role": "user", "content": "beta cooking"})

    results = await a.asearch("hiking", k=5)
    assert all(r.message.session_id == "a" for r in results)


async def test_hybrid_includes_recent_and_dedupes(store, embedder, vectors):
    mem = HybridMemory("s1", store, embedder, vectors, window_size=2, top_k=3)
    await mem.aadd({"role": "user", "content": "I am planning a trip to the Alps"})
    for i in range(4):
        await mem.aadd({"role": "user", "content": f"unrelated chatter {i}"})

    ctx = await mem.aget_context(query="Alps trip planning")
    contents = [m.content for m in ctx]
    # The relevant old message resurfaces via retrieval...
    assert any("Alps" in c for c in contents)
    # ...and the recent window is present.
    assert any("unrelated chatter 3" in c for c in contents)
    # No duplicate ids.
    ids = [m.id for m in ctx]
    assert len(ids) == len(set(ids))


async def test_areindex_makes_buffer_history_searchable(embedder, vectors):
    """Turns added under a non-semantic mode become retrievable after areindex —
    the mechanism that makes live mode-swapping seamless."""
    mem = OpenMemory()
    mem._embedder = embedder  # inject offline fakes instead of real providers
    mem._vectors = vectors

    buf = mem.session("live", strategy="buffer")
    await buf.aadd({"role": "user", "content": "my passport number is QX-7"})
    await buf.aadd({"role": "user", "content": "I like jazz music"})

    # Before reindexing, nothing is in the vector index.
    vec = mem.session("live", strategy="vector")
    assert await vec.asearch("passport", k=3) == []

    indexed = await mem.areindex("live")
    assert indexed == 2

    results = await vec.asearch("what is my passport number", k=1)
    assert results and "QX-7" in results[0].message.content
    await mem.aclose()
