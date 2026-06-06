# open-memory

Pluggable **context-management strategies** for LLM applications, behind one async,
session-scoped interface. Start naive (a list of dicts) and graduate to vector, hybrid,
or hierarchical memory **without changing your call sites**. Output is always clean,
OpenAI-compliant `[{"role", "content"}]` — internal bookkeeping fields are stored,
queryable, and filtered out at serialization time.

## Why

Every LLM app has to decide *what context to feed the model next turn*. The right answer
ranges from "keep everything" to "semantic retrieval over a vector DB" to "self-paging
hierarchical memory." open-memory makes those swappable strategies that share one API,
one message model, and one notion of sessions.

## Install

```bash
pip install open-memory                # core: OpenAI-compatible APIs + Qdrant
pip install "open-memory[litellm]"     # 100+ providers via LiteLLM
pip install "open-memory[local-llm]"   # offline embeddings/LLM (sentence-transformers, transformers, torch)
pip install "open-memory[all]"         # everything
```

The core install is intentionally lightweight. Heavy local ML deps (`torch`, etc.) only
arrive with the `local-llm` extra.

## Quick start

```python
from openmemory import OpenMemory

mem = OpenMemory()                                   # in-memory, buffer by default
chat = mem.session("user-42", strategy="hybrid")

await chat.aadd({"role": "user", "content": "I love hiking in the Alps"})
await chat.aadd({"role": "assistant", "content": "Noted! The Alps are gorgeous."})

# Assemble context for the next call (recent turns + semantically relevant history)
messages = await chat.aget_openai_context(query="suggest a vacation")
# messages -> [{"role": "...", "content": "..."}]  ready for any OpenAI-compatible API
```

Synchronous wrappers exist for scripts/notebooks (`chat.add(...)`, `chat.get_context()`);
call the `a`-prefixed methods from inside an event loop.

## Strategies

| `strategy=` | Behavior | Needs |
|---|---|---|
| `buffer` | Full history, replayed verbatim. The "list of dicts." | — |
| `window` | Most recent N turns and/or a token budget; keeps the system message. | — |
| `vector` | Semantic retrieval over Qdrant. | embedder + Qdrant |
| `hybrid` | Recent window **∪** semantic retrieval, deduped, chronological output. | embedder + Qdrant |
| `sparse_hybrid` | BM25 lexical **∪** dense vector, fused via Reciprocal Rank Fusion (RRF). | embedder + Qdrant |
| `full_hybrid` | Recency window **∪** BM25 **∪** dense vector, three-way RRF. Best coverage. | embedder + Qdrant |
| `hierarchical` | MemGPT-style: token-bounded working context + LLM rolling summary + archival. | embedder + Qdrant + LLM |
| `summary` | *Scaffold* — rolling summary + recent buffer. | (planned) |
| `facts` | *Scaffold* — mem0-style fact extraction. | (planned) |
| `graph` | *Scaffold* — entity/relationship graph retrieval. | (planned) |

### Hybrid strategies at a glance

Four strategies compose multiple retrieval signals. The table below shows how each
pair differs — read row-by-column: "what does the row strategy have that the column
strategy lacks?"

|  | **`hybrid`** | **`sparse_hybrid`** | **`full_hybrid`** | **`hierarchical`** |
|---|---|---|---|---|
| **`hybrid`** | — | Has a recency guarantee; lacks keyword precision | `full_hybrid` adds BM25 on top; `hybrid` has no keyword channel | `hierarchical` bounds context by LLM eviction; `hybrid` grows unbounded |
| **`sparse_hybrid`** | Has keyword precision (BM25); no recency guarantee | — | `full_hybrid` adds a recency window; `sparse_hybrid` is pure retrieval with no temporal bias | `hierarchical` compresses via LLM rolling summary; `sparse_hybrid` retrieves all turns forever |
| **`full_hybrid`** | Adds BM25 keyword channel to `hybrid`; three-way RRF over all signals | Adds recency window to `sparse_hybrid`; temporal signal participates in RRF | — | `hierarchical` is token-budget-bounded with eviction; `full_hybrid` grows unbounded but has richer retrieval |
| **`hierarchical`** | Compresses old turns into a rolling LLM summary; `hybrid` keeps all turns in raw form | Uses LLM eviction + archival for bounded context; `sparse_hybrid` retains all turns without compression | Same LLM-eviction model; `full_hybrid` trades compression for richer three-way fusion | — |

**Rule of thumb:**
- Keyword-heavy queries (names, IDs, codes) → `sparse_hybrid` or `full_hybrid`
- Very long sessions where context budget matters → `hierarchical`
- Default chat assistant → `hybrid` or `full_hybrid`

Per-session overrides go straight to the strategy:

```python
mem.session("u1", strategy="window", n=50, token_budget=4000)
mem.session("u1", strategy="hybrid", window_size=10, top_k=8)
```

## Configuration

Set programmatically via `Config(...)` or through `OPENMEMORY_*` environment variables:

```python
from openmemory import OpenMemory, Config

cfg = Config(
    embedder_provider="litellm",        # "openai" | "litellm" | "local"
    embedding_model="text-embedding-3-small",
    llm_provider="openai",              # used by hierarchical's summarizer
    qdrant_url="http://localhost:6333", # None => in-memory Qdrant
    session_store="sqlite",             # "memory" | "sqlite"
)
mem = OpenMemory(cfg)
```

## Qdrant via Docker

```bash
docker compose up -d                    # starts qdrant on :6333 / :6334
export OPENMEMORY_QDRANT_URL=http://localhost:6333
```

Without a URL, an embedded in-memory Qdrant is used — perfect for tests and quick demos.

## Message model

Each `Message` carries OpenAI wire fields (`role`, `content`, `name`, `tool_calls`,
`tool_call_id`) plus internal fields you can filter and rank on: `id`, `session_id`,
`timestamp`, `tags`, `importance`, `metadata`, `embedding`. `to_openai_messages(...)`
emits only the wire fields.

## Development

```bash
uv sync --extra dev
uv run pytest          # hermetic: in-memory Qdrant + SQLite tmp, no network
uv run ruff check
uv run mypy src/
```

## Roadmap

- Implement the `summary`, `facts` (mem0-style), and `graph` strategies (interfaces are
  already in place via `BaseMemory` and `GraphStore`).
- Restart-durable rolling summaries for hierarchical memory.
- Cross-encoder reranking as a post-retrieval step on top of `sparse_hybrid` / `full_hybrid`.
- Persistent BM25 index (SQLite FTS5 backend) so lexical recall survives restarts.
