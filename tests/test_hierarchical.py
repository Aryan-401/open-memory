from __future__ import annotations

from openmemory.llm.summarizer import Summarizer
from openmemory.strategies.hierarchical import HierarchicalMemory


async def test_eviction_creates_summary_and_archival_is_retrievable(
    store, embedder, vectors, llm
):
    mem = HierarchicalMemory(
        "s1",
        store,
        embedder,
        vectors,
        Summarizer(llm),
        working_context_tokens=20,  # tiny budget to force eviction
        top_k=3,
    )

    await mem.aadd({"role": "user", "content": "remember my passport number is ABC123"})
    for i in range(8):
        await mem.aadd({"role": "user", "content": f"chatting about topic number {i}"})

    # Eviction should have summarized the oldest turns.
    assert llm.calls > 0
    assert mem._summary != ""

    ctx = await mem.aget_context(query="what is my passport number")
    contents = [m.content for m in ctx]
    # Summary block is present as a system message.
    assert any("Summary of earlier conversation" in c for c in contents)
    # The evicted passport fact is retrievable from archival via the query.
    assert any("ABC123" in c for c in contents)


async def test_clear_resets_summary(store, embedder, vectors, llm):
    mem = HierarchicalMemory(
        "s1", store, embedder, vectors, Summarizer(llm), working_context_tokens=10
    )
    for i in range(6):
        await mem.aadd({"role": "user", "content": f"message {i} with some words"})
    await mem.aclear()
    assert mem._summary == ""
    assert mem._summarized_count == 0
    assert await mem.aget_context() == []
