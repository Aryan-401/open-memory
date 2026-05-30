# Examples

One runnable script per fully-implemented strategy, plus a live chat REPL that exposes
the internal machinery. Each script shows what context is assembled and why, then calls
your chosen LLM with it and prints the reply.

---

## Setup

```bash
pip install "open-memory[examples]"
```

Copy `.env.example` → `.env` and fill in your key, or export the variables directly:

```bash
export OPENMEMORY_EXAMPLE_PROVIDER=groq   # openai | claude | gemini | groq | huggingface | litellm
export GROQ_API_KEY=gsk_...
```

| `PROVIDER` | Chat model | Key variable | Embeddings |
|---|---|---|---|
| `openai` | `gpt-4o-mini` (native) | `OPENAI_API_KEY` | OpenAI |
| `claude` | `claude-3-5-haiku` (via LiteLLM) | `ANTHROPIC_API_KEY` | local |
| `gemini` | `gemini-1.5-flash` (via LiteLLM) | `GEMINI_API_KEY` | Gemini |
| `groq` | `llama-3.1-8b-instant` (via LiteLLM) | `GROQ_API_KEY` | local |
| `huggingface` | `SmolLM2-360M-Instruct` (local) | — | local |
| `litellm` | `$OPENMEMORY_LLM_MODEL` | provider's own | local |

> "local" embeddings use `sentence-transformers/all-MiniLM-L6-v2`, downloaded on first run.
> Vector/hybrid/hierarchical use an **in-memory Qdrant** by default (no Docker).
> For a persistent instance: `docker compose up -d` and set `OPENMEMORY_QDRANT_URL`.

---

## Scripts

```bash
python examples/01_buffer.py        # full-history replay
python examples/02_window.py        # recent-N turns only
python examples/03_vector.py        # semantic retrieval (Qdrant)
python examples/04_hybrid.py        # recency + semantic, merged
python examples/05_hierarchical.py  # working context + summary + archival
python examples/06_live_chat.py     # interactive REPL — swap modes live
python examples/07_summary.py       # rolling LLM summary + recent buffer
python examples/08_facts.py         # atomic fact extraction + deduplication
python examples/09_graph.py         # knowledge graph (networkx triplets)
```

---

## Live chat (`06_live_chat.py`) — internal steps

[`06_live_chat.py`](06_live_chat.py) is an interactive REPL where you type messages,
get replies from the LLM, and **see in grey text exactly what open-memory does on every
step**. You can switch the active memory strategy mid-conversation with `/mode` and watch
the behaviour change immediately.

The prompt shows the active strategy and, in semantic modes, the active embed mode:

```text
(buffer) you>                  # buffer/window — no embed mode shown
(hybrid,embed=per_message) you>  # semantic — embed mode visible
(hybrid,embed=paired) you>       # after /embed-mode paired
```

---

```text
▶ strategy=buffer  provider=groq  model=groq/llama-3.1-8b-instant
  internal steps shown in grey  —  /help for commands

(buffer) you> my flight is on June 14th
  ⚙  [buffer] load all messages from session store → 0 messages
  ⚙  [buffer] strategy: no filtering, no ranking — return everything verbatim
  ▸ context ready: 0 messages [1ms]
  ⚙  send 1 messages to groq/llama-3.1-8b-instant...
  ⚙  response received [643ms]

[buffer] assistant: Got it — June 14th noted!

  ⚙  append 2 messages to session store

(buffer) you> /mode hybrid
  ⚙  switching to a semantic mode — re-indexing history into Qdrant...
  ⚙    1. load all stored messages from session store
  ⚙    2. embed each message text using the configured embedder
  ⚙    3. upsert vectors to Qdrant (keyed by message id — safe to repeat)
  ⚙    done: 2 messages embedded and upserted [collection=openmemory, session=live]
  → switched buffer → hybrid

(hybrid) you> when is my flight?
  ⚙  [hybrid] session store has 2 stored messages
  ⚙  [hybrid] ── step 1/3: semantic branch ──────────────────
  ⚙  [hybrid] embed query: 'when is my flight?'
  ⚙  [hybrid] Qdrant search [collection=openmemory, top-k=5] [38ms]
  ⚙  [hybrid] retrieved 2 semantic matches (scores: 0.812, 0.491)
  ⚙  [hybrid] ── step 2/3: recency branch ───────────────────
  ⚙  [hybrid] fetch last 20 messages from session store
  ⚙  [hybrid] window: 2 most recent messages
  ⚙  [hybrid] ── step 3/3: merge ─────────────────────────────
  ⚙  [hybrid] deduplicate: 2 semantic + 2 window = 2 unique (2 already in window, dropped)
  ⚙  [hybrid] sort merged set chronologically
  ▸ context ready: 2 messages [42ms]
  ⚙  send 3 messages to groq/llama-3.1-8b-instant...
  ⚙  response received [511ms]

[hybrid] assistant: Your flight is on June 14th.
```

---

### What each method does internally

#### `say(text)` — handles a typed message

1. **Assemble context** using the active strategy (details per strategy below).
2. **Append** the current user message to the context list (the strategy's output is
   *background* context; the live message is always the final item sent to the model).
3. **Call the LLM** with the assembled context.
4. **Persist the exchange** — the user message and the assistant reply are stored:
   - Non-semantic modes (`buffer`, `window`): appended to the session store only.
   - Semantic modes (`vector`, `hybrid`, `hierarchical`): embedded and upserted to Qdrant,
     then appended to the session store.
5. **Eviction check** (hierarchical only): if the working context now exceeds its token
   budget, the overflow messages are sent to the LLM summarizer and the rolling summary
   is updated; the evicted messages stay in Qdrant for future archival retrieval.

---

#### Context assembly, by strategy

**`buffer`**
1. Load the full message list from the session store.
2. Return it unchanged — no filtering, no ranking.

**`window`**
1. Load the full message list from the session store.
2. Preserve any leading `system` message unconditionally.
3. Walk the remaining messages newest-first:
   - Stop at count `n` (if set).
   - Stop when adding the next message would exceed `token_budget` (if set, counted with
     tiktoken; falls back to a character heuristic for unknown models).
4. Return the system message (if any) + the kept tail.

**`vector`**
1. Embed the query text using the configured `Embedder`
   (OpenAI API call, LiteLLM call, or a local sentence-transformers model on CPU thread).
2. Run a filtered approximate-nearest-neighbour (ANN) search in Qdrant:
   `query_filter = {session_id: "live"}` so no other session's vectors are touched.
3. Take the top-`k` hits by cosine similarity score.
4. Re-sort the results chronologically (Qdrant returns by relevance; the model reads
   them in time order for natural coherence).

**`hybrid`**  ← *recommended default*
1. **Semantic branch**: embed the query → Qdrant ANN search → top-`k` results.
2. **Recency branch**: load the last `window_size` messages from the session store.
3. **Merge**: collect all unique messages by id (any message present in both branches
   is counted once). Sort the merged set chronologically.

**`summary`**
1. Load all stored messages from the session store.
2. Split off any leading `system` message.
3. The buffer tail (last `buffer_size` messages) is kept verbatim. Everything older is
   covered by the rolling summary (produced lazily at `aadd` time, not on `aget_context`).
4. Assemble: `system (if any) → summary block (if any) → buffer`.

**`facts`**
1. Embed the query → ANN search in the dedicated `{collection}_facts` Qdrant collection
   (separate from the main message store, holds only extracted fact strings).
2. Return the top-k matching facts as a single `system` message:
   `"Known facts:\n- fact1\n- fact2\n…"`
3. Without a query, return all known facts (from the in-process list).

**`graph`**
1. Extract entity names from the query (heuristic: capitalised words).
2. Traverse the session's `networkx.DiGraph` up to `hops` steps from each entity.
3. Return the relevant `(subject, predicate, object)` triplets as a single `system` message:
   `"Knowledge graph context:\n- Alice works_with Bob\n…"`
4. Without a query, return all edges in the graph.

**`hierarchical`** (MemGPT-style)
1. Load all stored messages. Split off any leading `system` message.
2. Walk the remaining body tail-first, fitting messages within `working_context_tokens`.
   The messages that fit are the **working tail**; the rest are **archived**.
3. Determine summary state: if eviction has happened before, a rolling summary block
   exists summarising all archived messages.
4. Embed the query → Qdrant archival search → top-`k` hits not already in the working
   tail.
5. Assemble final context:
   `system (if any) → summary block (if any) → archival hits → working tail`

---

#### `/embed-mode <name>` — switch embedding strategy (semantic modes only)

Two options:

| Mode | What gets embedded | When to use |
|---|---|---|
| `per_message` | Each message uses its own content | Default; precise retrieval, lower noise |
| `paired` | User message vector includes the assistant reply: `"User: …\nAssistant: …"` | Better recall — the assistant's paraphrase adds semantic surface area |

Switching embed mode triggers an automatic re-index so all existing turns are re-embedded
with the new strategy. Both modes keep user and assistant as **separate Message records**;
only the text used to compute the user message's vector changes.

```text
(hybrid,embed=per_message) you> /embed-mode paired
  ▸ re-indexing history with new embed mode so existing turns use it...
  ⚙    embed_mode=paired: user message vectors enriched with assistant reply text
  ⚙    done: 4 vectors updated [collection=openmemory]
  → embed-mode per_message → paired
```

---

#### `/mode <name>` — switch active strategy

1. If switching to a semantic mode (`vector`, `hybrid`, `hierarchical`):
   a. Load all messages from the session store.
   b. Embed each one in batch using the configured `Embedder`.
   c. Upsert all vectors to Qdrant, keyed by `message.id` (idempotent — safe to run
      multiple times; any message already indexed is simply overwritten with the same
      vector).
2. Set the active mode; the prompt prefix updates immediately.

All strategies share the same `session_id` and therefore the same underlying session
store, so no history is lost when switching.

---

#### `/context [query]` — preview context without calling the LLM

Calls `aget_context(query=query)` on the current session and prints each message in the
result. Useful for verifying what would actually be sent to the model. Passing a query
activates semantic ranking (relevant for `vector`, `hybrid`, `hierarchical`).

---

#### `/search <query>` — raw semantic search

Calls `asearch(query, k=5)` and prints the top results with their cosine similarity
scores. Only works in semantic modes (`vector`, `hybrid`, `hierarchical`). Use this to
verify that a particular fact is indexed and retrievable.

---

#### `/history` — full stored conversation

Reads directly from the session store (bypassing the active strategy) and prints every
message in insertion order. Shows the raw source of truth regardless of which strategy
is active.

---

#### `/clear` — wipe the session

1. Calls `aclear()` on the current strategy, which:
   - Deletes all messages for this `session_id` from the session store.
   - Deletes all vectors for this `session_id` from Qdrant (filtered delete, not a
     full collection drop — other sessions are unaffected).
2. Resets the hierarchical rolling summary and eviction counter if present.

---

---

### `summary` — rolling summary + verbatim recent buffer

**What it does** — LangChain ConversationSummaryBufferMemory pattern, two tiers, no
vector store required:

1. Append new messages to the session store.
2. If the buffer (`buffer_size` recent messages) has been exceeded, fold the overflow into
   a rolling summary via an LLM call (`Summarizer`).
3. `aget_context` returns: `[system] + [summary-as-system] + [buffer]`.

**Separate LLM**: set `OPENMEMORY_SUMMARY_LLM_PROVIDER` / `OPENMEMORY_SUMMARY_LLM_MODEL`
to use a different (e.g. cheaper/faster) model for compression than for chat.

---

### `facts` — atomic fact extraction + deduplication (mem0-style)

**What it does** — instead of storing raw turns, an LLM distils each exchange into a
list of durable atomic facts and stores them in a dedicated Qdrant collection:

1. Append raw messages to the session store (for `/history`).
2. Call the extraction LLM with the new transcript → JSON array of fact strings.
3. For each candidate fact: embed it and run a similarity check against existing facts.
   If cosine score ≥ `dedup_threshold` → skip (already known). Otherwise upsert to
   Qdrant (`{collection}_facts` — separate from the main message store).
4. `aget_context(query=...)` → semantic search over facts → return as one system message
   listing the relevant facts. Without a query, return all known facts.

**Separate LLM**: set `OPENMEMORY_FACTS_LLM_PROVIDER` / `OPENMEMORY_FACTS_LLM_MODEL`.

---

### `graph` — knowledge graph (networkx, Neo4j/Graphiti planned)

**What it does** — an LLM extracts (subject, predicate, object) triplets from each turn
and accumulates them into a session-scoped `networkx.DiGraph`. On query, entity names
are extracted from the query text (heuristic: capitalised words) and the graph is
traversed up to `hops` steps to find related triplets, returned as a system message:

1. Append raw messages to the session store (for `/history`).
2. Call the extraction LLM → JSON array of `[subject, predicate, object]` triplets.
3. Add new edges to the in-process `NetworkxGraphStore` (first-write-wins per edge).
4. `aget_context(query=...)` → extract entities → `neighborhood(entities, hops=N)` →
   return subgraph as a system message. Without a query, returns all edges.

**Future backend**: the `GraphStore` protocol is stable — a `Neo4jGraphStore` or
`GraphitiStore` can slot in behind it without changing `GraphMemory`.

**Separate LLM**: set `OPENMEMORY_GRAPH_LLM_PROVIDER` / `OPENMEMORY_GRAPH_LLM_MODEL`.

---

## Strategy cheat-sheet

| Strategy | What it feeds the model | LLM on add? | Vector index? | When to use |
|---|---|---|---|---|
| `buffer` | Everything, always | No | No | Short chats, debugging |
| `window` | The most recent N turns | No | No | Fixed-length context, bounded cost |
| `vector` | The most semantically relevant turns | No | Yes (messages) | Query-driven retrieval |
| `hybrid` | Recent turns + semantically relevant turns | No | Yes (messages) | **Default for most assistants** |
| `hierarchical` | Recent tail + summary + archival hits | Yes (summary) | Yes (messages) | Very long sessions, agents |
| `summary` | Summary block + recent buffer | Yes (summary) | No | Long sessions, no retrieval needed |
| `facts` | Relevant extracted facts | Yes (extraction) | Yes (facts) | Preference tracking, personal assistants |
| `graph` | Relevant graph triplets | Yes (extraction) | No | Relationship-heavy domains, org charts |
