from __future__ import annotations

import pytest

from openmemory import Config, Message, OpenMemory, to_openai_messages
from openmemory.strategies.facts import FactExtractionMemory
from openmemory.strategies.graph import GraphMemory
from openmemory.strategies.summary import SummaryMemory


def test_to_openai_messages_strips_internal_fields():
    msg = Message(
        role="user",
        content="hello",
        tags=["greeting"],
        importance=0.9,
        metadata={"x": 1},
    )
    wire = to_openai_messages([msg])
    assert wire == [{"role": "user", "content": "hello"}]
    for forbidden in ("id", "session_id", "timestamp", "tags", "importance", "metadata"):
        assert forbidden not in wire[0]


def test_tool_fields_preserved_on_wire():
    msg = Message(role="tool", content="result", name="search", tool_call_id="call_1")
    wire = msg.to_openai()
    assert wire["name"] == "search"
    assert wire["tool_call_id"] == "call_1"


async def test_facade_buffer_roundtrip_no_credentials():
    mem = OpenMemory()  # buffer default, in-memory store, no API keys needed
    chat = mem.session("u1", strategy="buffer")
    await chat.aadd({"role": "user", "content": "hi"})
    out = await chat.aget_openai_context()
    assert out == [{"role": "user", "content": "hi"}]


async def test_facade_sessions_do_not_leak():
    mem = OpenMemory()
    a = mem.session("a", strategy="window")
    b = mem.session("b", strategy="window")
    await a.aadd({"role": "user", "content": "secret-a"})
    await b.aadd({"role": "user", "content": "secret-b"})
    assert [m.content for m in await a.aget_context()] == ["secret-a"]
    assert [m.content for m in await b.aget_context()] == ["secret-b"]


def test_sync_wrapper_works_outside_loop():
    mem = OpenMemory()
    chat = mem.session("sync", strategy="buffer")
    chat.add({"role": "user", "content": "sync hello"})
    assert chat.get_openai_context() == [{"role": "user", "content": "sync hello"}]


async def test_sync_wrapper_raises_inside_loop():
    mem = OpenMemory()
    chat = mem.session("sync2", strategy="buffer")
    with pytest.raises(RuntimeError, match="running event loop"):
        chat.add({"role": "user", "content": "x"})


async def test_all_strategies_selectable_via_client(store, embedder, llm, vectors):
    """All eight strategies can be instantiated via OpenMemory.session()."""
    from openmemory.config import Config
    cfg = Config(
        embedder_provider="openai",
        llm_provider="openai",
        session_store="memory",
    )
    mem = OpenMemory(cfg)
    # Wire shared resources manually to avoid API calls.
    mem._store = store
    mem._embedder = embedder
    mem._llm = llm
    mem._vectors = vectors
    mem._fact_vectors = vectors  # reuse for simplicity in this test

    for strategy in ["buffer", "window", "vector", "hybrid", "hierarchical",
                     "summary", "facts", "graph"]:
        sess = mem.session("check", strategy=strategy)  # type: ignore[arg-type]
        assert sess is not None


def test_config_env_prefix(monkeypatch):
    monkeypatch.setenv("OPENMEMORY_WINDOW_SIZE", "7")
    cfg = Config()
    assert cfg.window_size == 7
