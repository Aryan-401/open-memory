# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra dev          # install dev dependencies
uv run pytest                # run all tests (hermetic: no network, no Docker required)
uv run pytest tests/test_buffer_window.py  # run a single test file
uv run pytest -k "test_name" # run a single test by name
uv run ruff check            # lint
uv run mypy src/             # type-check
```

For running examples (requires API keys):
```bash
uv sync --extra examples
OPENAI_API_KEY=... uv run python examples/01_buffer.py
```

For Docker-backed Qdrant (vector/hybrid/hierarchical strategies):
```bash
docker compose up -d   # starts Qdrant on :6333 / :6334
export OPENMEMORY_QDRANT_URL=http://localhost:6333
```

## Architecture

The library is structured around **strategies** — swappable context-management policies that all implement `BaseMemory` (`src/openmemory/core/base.py`).

**Entry point**: `OpenMemory` (`client.py`) is the facade. Call `mem.session(id, strategy=...)` to get a `Session`. The client lazily builds shared resources (embedder, LLM, Qdrant client, summarizer) the first time a strategy that needs them is created.

**Session** (`session.py`) is a thin wrapper around a `BaseMemory` instance that carries `session_id` and `metadata`. It delegates all calls through.

**BaseMemory** (`core/base.py`) defines the contract: async-first (`aadd`, `aget_context`, `asearch`, `aclear`). Sync wrappers (`add`, `get_context`, etc.) live on the base class via `run_sync` (`_sync.py`). Calling a sync wrapper from inside a running event loop raises `RuntimeError` — use the `a`-prefixed async method instead.

**Strategies** (`strategies/`):
- `BufferMemory` — full history replayed verbatim; no dependencies
- `WindowMemory` — last N turns within a tiktoken token budget
- `VectorMemory` — semantic retrieval from Qdrant only
- `HybridMemory` — recent window union'd with semantic retrieval, deduped (recommended default)
- `HierarchicalMemory` — MemGPT-style: token-bounded working-context tail + rolling summary (evicted turns folded by `Summarizer`) + Qdrant archival. Summary is held in-process and is not restart-durable yet.
- `SummaryMemory`, `FactExtractionMemory`, `GraphMemory` — scaffolds, not yet implemented

**Message model** (`core/models.py`): `Message` carries OpenAI wire fields (`role`, `content`, `name`, `tool_calls`, `tool_call_id`) plus internal fields (`id`, `session_id`, `timestamp`, `tags`, `importance`, `metadata`, `embedding`). `to_openai()` / `to_openai_messages()` strips internal fields before returning to callers.

**Storage layer**:
- `SessionStore` (`storage/session_store.py`) — ordered message log per session; backends are `InMemorySessionStore` and `SQLiteSessionStore`
- `QdrantVectorStore` (`storage/qdrant_store.py`) — single Qdrant collection shared across all sessions, filtered by `session_id` payload. `url=None` → in-process Qdrant (`:memory:`), no Docker needed

**Provider backends** (all lazily imported to keep the core install lightweight):
- `embeddings/`: `openai`, `litellm`, `local` (sentence-transformers)
- `llm/`: `openai`, `litellm`, `local` (HuggingFace transformers)
- Selected by `Config.embedder_provider` / `Config.llm_provider`

**Config** (`config.py`): `pydantic-settings` — all fields readable from `OPENMEMORY_*` env vars or a `.env` file. Key settings: `qdrant_url`, `session_store` (`"memory"` | `"sqlite"`), `embedder_provider`, `llm_provider`, `window_size`, `retrieval_top_k`, `embed_mode` (`"per_message"` | `"paired"`).

## Tests

Tests are fully hermetic — `conftest.py` provides `FakeEmbedder` (deterministic bag-of-words cosine similarity), `FakeLLM` (echo summarizer), `InMemorySessionStore`, and in-memory Qdrant. No network or Docker needed. `asyncio_mode = "auto"` is set in `pyproject.toml` so async test functions don't need `@pytest.mark.asyncio`.
