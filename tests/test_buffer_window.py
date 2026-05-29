from __future__ import annotations

from openmemory import Message
from openmemory.strategies.buffer import BufferMemory
from openmemory.strategies.window import WindowMemory


async def test_buffer_returns_full_history_in_order(store):
    mem = BufferMemory("s1", store)
    await mem.aadd({"role": "user", "content": "one"})
    await mem.aadd({"role": "assistant", "content": "two"})
    await mem.aadd({"role": "user", "content": "three"})

    ctx = await mem.aget_context()
    assert [m.content for m in ctx] == ["one", "two", "three"]


async def test_window_caps_by_count_and_keeps_system(store):
    mem = WindowMemory("s1", store, n=2)
    await mem.aadd({"role": "system", "content": "be nice"})
    for i in range(5):
        await mem.aadd({"role": "user", "content": f"msg{i}"})

    ctx = await mem.aget_context()
    # system preserved + last 2 user messages
    assert ctx[0].role == "system"
    assert [m.content for m in ctx[1:]] == ["msg3", "msg4"]


async def test_window_token_budget(store):
    mem = WindowMemory("s1", store, n=None, token_budget=12)
    for i in range(10):
        await mem.aadd({"role": "user", "content": f"word{i}"})

    ctx = await mem.aget_context()
    # Each message ~1 token + 4 overhead = 5 tokens; budget 12 fits 2 messages.
    assert len(ctx) == 2
    assert ctx[-1].content == "word9"


async def test_get_context_limit_override(store):
    mem = WindowMemory("s1", store, n=100)
    for i in range(5):
        await mem.aadd({"role": "user", "content": f"m{i}"})
    ctx = await mem.aget_context(limit=1)
    assert [m.content for m in ctx] == ["m4"]


async def test_buffer_stamps_session_id(store):
    mem = BufferMemory("abc", store)
    await mem.aadd(Message(role="user", content="hi"))
    ctx = await mem.aget_context()
    assert ctx[0].session_id == "abc"
