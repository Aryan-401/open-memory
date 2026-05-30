"""Tests for SummaryMemory."""

from __future__ import annotations

import pytest

from openmemory.llm.summarizer import Summarizer
from openmemory.strategies.summary import SummaryMemory


async def _mem(store, llm, buffer_size: int = 4) -> SummaryMemory:
    return SummaryMemory(
        "s-test", store, Summarizer(llm), buffer_size=buffer_size
    )


async def test_buffer_only_no_summary(store, llm):
    """When messages fit within the buffer no LLM call is made."""
    mem = await _mem(store, llm, buffer_size=6)
    for i in range(3):
        await mem.aadd({"role": "user", "content": f"msg {i}"})
        await mem.aadd({"role": "assistant", "content": f"reply {i}"})

    assert llm.calls == 0
    ctx = await mem.aget_context()
    assert len(ctx) == 6
    assert mem._summary == ""


async def test_overflow_triggers_compression(store, llm):
    """When buffer overflows the older messages are folded into a summary."""
    mem = await _mem(store, llm, buffer_size=2)
    for i in range(4):
        await mem.aadd({"role": "user", "content": f"turn {i}"})
        await mem.aadd({"role": "assistant", "content": f"reply {i}"})

    assert llm.calls > 0
    assert mem._summary != ""
    ctx = await mem.aget_context()
    roles = [m.role for m in ctx]
    # Should contain a summary system message plus the verbatim buffer
    summary_msgs = [m for m in ctx if m.metadata.get("openmemory_kind") == "summary"]
    assert len(summary_msgs) == 1
    body_msgs = [m for m in ctx if m.role in {"user", "assistant"}]
    assert len(body_msgs) <= 2


async def test_summary_present_as_system_message(store, llm):
    """The rolling summary appears as a system-role message in context."""
    mem = await _mem(store, llm, buffer_size=2)
    for i in range(3):
        await mem.aadd({"role": "user", "content": f"fact {i}"})
        await mem.aadd({"role": "assistant", "content": f"ack {i}"})

    ctx = await mem.aget_context()
    summary_msgs = [m for m in ctx if m.metadata.get("openmemory_kind") == "summary"]
    assert len(summary_msgs) == 1
    assert "Summary" in summary_msgs[0].content


async def test_system_message_preserved(store, llm):
    """A leading system message is always kept and not counted toward the buffer."""
    mem = await _mem(store, llm, buffer_size=2)
    await mem.aadd({"role": "system", "content": "You are a helpful assistant."})
    for i in range(4):
        await mem.aadd({"role": "user", "content": f"q {i}"})
        await mem.aadd({"role": "assistant", "content": f"a {i}"})

    ctx = await mem.aget_context()
    assert ctx[0].role == "system"
    assert ctx[0].metadata.get("openmemory_kind") != "summary"


async def test_asearch_returns_empty(store, llm):
    """SummaryMemory has no vector index; asearch always returns []."""
    mem = await _mem(store, llm)
    await mem.aadd({"role": "user", "content": "something"})
    results = await mem.asearch("something")
    assert results == []


async def test_clear_resets_state(store, llm):
    """aclear wipes both the store and the in-process summary state."""
    mem = await _mem(store, llm, buffer_size=2)
    for i in range(4):
        await mem.aadd({"role": "user", "content": f"x {i}"})
        await mem.aadd({"role": "assistant", "content": f"y {i}"})

    await mem.aclear()
    assert mem._summary == ""
    assert mem._summarized_count == 0
    ctx = await mem.aget_context()
    assert ctx == []


async def test_sessions_isolated(store, llm):
    """Two SummaryMemory instances with different session_ids don't share state."""
    m1 = SummaryMemory("sess-1", store, Summarizer(llm), buffer_size=10)
    m2 = SummaryMemory("sess-2", store, Summarizer(llm), buffer_size=10)
    await m1.aadd({"role": "user", "content": "session one"})
    await m2.aadd({"role": "user", "content": "session two"})

    ctx1 = await m1.aget_context()
    ctx2 = await m2.aget_context()
    assert len(ctx1) == 1
    assert len(ctx2) == 1
    assert ctx1[0].content != ctx2[0].content
