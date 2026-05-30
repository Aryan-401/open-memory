"""Live chat REPL — swap memory modes on demand and see the internal machinery.

Every turn prints a step-by-step trace (in grey) of what open-memory does under the
hood: which store it reads, how the query is embedded, what Qdrant returns, how context
is assembled, how long each step took, and when hierarchical memory evicts and summarises.
Switch modes mid-conversation with /mode to feel the difference in real time.

    pip install "open-memory[examples]"
    export OPENMEMORY_EXAMPLE_PROVIDER=groq
    export GROQ_API_KEY=gsk_...
    python examples/06_live_chat.py

Commands:
    /mode <buffer|window|vector|hybrid|hierarchical>  switch active strategy
    /context [query]   show the context that would be sent (no LLM call)
    /search <query>    semantic search over stored history
    /history           print the full stored conversation
    /clear             wipe the session
    /help              show this help
    /quit              exit
"""

from __future__ import annotations

import asyncio
import time

from _common import PROVIDER, banner, build_config

from openmemory import OpenMemory
from openmemory.core.models import Message, to_openai_messages
from openmemory.core.tokens import count_messages_tokens
from openmemory.llm.base import build_llm
from openmemory.strategies.buffer import BufferMemory
from openmemory.strategies.facts import FactExtractionMemory
from openmemory.strategies.graph import GraphMemory
from openmemory.strategies.hierarchical import HierarchicalMemory
from openmemory.strategies.hybrid import HybridMemory
from openmemory.strategies.summary import SummaryMemory
from openmemory.strategies.vector import VectorMemory
from openmemory.strategies.window import WindowMemory

SESSION_ID = "live"
SEMANTIC = {"vector", "hybrid", "hierarchical"}
LLM_MODES = {"summary", "facts", "graph"}  # modes that call LLM on every aadd
MODES = ["buffer", "window", "vector", "hybrid", "hierarchical", "summary", "facts", "graph"]
EMBED_MODES = ["per_message", "paired"]

# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------

_COMMANDS = [
    "/mode", "/embed-mode", "/context", "/search",
    "/history", "/clear", "/help", "/quit",
]

# Commands that take a fixed set of arguments get suggestions for those too.
_COMMAND_ARGS: dict[str, list[str]] = {
    "/mode": MODES,
    "/embed-mode": EMBED_MODES,
    "/embed_mode": EMBED_MODES,  # alias accepted by the dispatcher
}


class _TabCompleter:
    """readline-based tab completion for /commands.

    Handles two levels:
      - partial command name  → complete to a full command (e.g. "/mo" → "/mode ")
      - command + partial arg → complete to a known argument value
                                (e.g. "/mode buf" → "buffer")

    Caches the match list on state=0 so successive Tab presses are consistent.
    """

    def __init__(self) -> None:
        self._matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        try:
            import readline
            if state == 0:
                self._matches = self._build(readline.get_line_buffer())
            return self._matches[state]
        except IndexError:
            return None
        except Exception:
            return None

    @staticmethod
    def _build(line: str) -> list[str]:
        if not line.startswith("/"):
            return []
        parts = line.split()
        trailing = line.endswith(" ")
        if not parts:
            return []

        if len(parts) == 1 and not trailing:
            # Still typing the command name itself.
            partial = parts[0]
            return [c + " " for c in _COMMANDS if c.startswith(partial)]

        # Command is complete; suggest argument values if known.
        cmd = parts[0]
        partial = "" if trailing else parts[-1]
        return [m for m in _COMMAND_ARGS.get(cmd, []) if m.startswith(partial)]


def _setup_readline() -> None:
    """Wire up tab completion via readline (no-op if readline is unavailable)."""
    try:
        import readline
        completer = _TabCompleter()
        readline.set_completer(completer.complete)
        # Only split on space/tab so "/embed-mode" stays as one token.
        readline.set_completer_delims(" \t")
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass  # readline not available on Windows; graceful degradation


# ANSI colour codes (terminals that don't support them will show plain text)
DIM    = "\033[2m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

HELP = """\
commands:
  /mode <name>        switch strategy:
                        buffer | window | vector | hybrid | hierarchical
                        summary | facts | graph
  /embed-mode <name>  switch embedding mode (semantic modes): per_message | paired
  /context [query]    show context that would be sent to the model (no LLM call)
  /search <query>     semantic search (vector/hybrid/hierarchical/facts modes)
  /history            print the full stored conversation
  /clear              wipe the session
  /help               show this help
  /quit               exit"""


def step(msg: str) -> None:
    """Grey trace line — shows internal machinery."""
    print(f"{DIM}  ⚙  {msg}{RESET}")


def info(msg: str) -> None:
    """Cyan summary line — highlights a key result."""
    print(f"{CYAN}  ▸ {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"{YELLOW}  ⚠  {msg}{RESET}")


class LiveChat:
    def __init__(self) -> None:
        self.cfg = build_config()
        self.mem = OpenMemory(self.cfg)
        self.llm = build_llm(self.cfg)
        self.mode = "buffer"
        self.embed_mode = self.cfg.embed_mode  # "per_message" or "paired"
        # One cached session per (strategy, embed_mode) — they share the same session_id
        # and therefore the same underlying store; hierarchical keeps its rolling-summary
        # state across turns because the instance is reused.
        self._sessions: dict[str, object] = {}

    def _session_key(self) -> str:
        return f"{self.mode}:{self.embed_mode}"

    def session(self, mode: str):  # type: ignore[no-untyped-def]
        key = self._session_key()
        if key not in self._sessions:
            self._sessions[key] = self.mem.session(
                SESSION_ID, strategy=mode, embed_mode=self.embed_mode
            )
        return self._sessions[key]

    # -----------------------------------------------------------------------
    # /mode — switch active strategy
    # -----------------------------------------------------------------------

    async def switch(self, mode: str) -> None:
        if mode not in MODES:
            print(f"  unknown mode {mode!r}; choose: {', '.join(MODES)}")
            return
        old = self.mode
        self.mode = mode
        col = self.cfg.qdrant_collection
        if mode in SEMANTIC:
            step("switching to a semantic mode — re-indexing history into Qdrant...")
            step("  1. load all stored messages from session store")
            step(f"  2. embed messages (embed_mode={self.embed_mode})")
            step("  3. upsert vectors to Qdrant (keyed by message id — safe to repeat)")
            try:
                n = await self.mem.areindex(SESSION_ID, embed_mode=self.embed_mode)
                step(f"  done: {n} messages embedded and upserted "
                     f"[collection={col}, session={SESSION_ID}]")
            except Exception as exc:
                warn(f"could not index history: {exc}")
        print(f"  → switched {old} → {mode}")

    # -----------------------------------------------------------------------
    # /embed-mode — switch embedding strategy
    # -----------------------------------------------------------------------

    async def switch_embed_mode(self, mode: str) -> None:
        if mode not in EMBED_MODES:
            print(f"  unknown embed-mode {mode!r}; choose: {', '.join(EMBED_MODES)}")
            return
        old = self.embed_mode
        self.embed_mode = mode
        print(f"  → embed-mode {old} → {mode}")
        if self.mode in SEMANTIC:
            info("re-indexing history with new embed mode so existing turns use it...")
            col = self.cfg.qdrant_collection
            step(f"  embed_mode={mode}: " + (
                "user message vectors enriched with assistant reply text"
                if mode == "paired"
                else "each message embedded from its own content only"
            ))
            try:
                n = await self.mem.areindex(SESSION_ID, embed_mode=self.embed_mode)
                step(f"  done: {n} vectors updated [collection={col}]")
            except Exception as exc:
                warn(f"could not re-index: {exc}")

    # -----------------------------------------------------------------------
    # Main turn handler — with full step-by-step trace
    # -----------------------------------------------------------------------

    async def say(self, text: str) -> None:
        sess = self.session(self.mode)
        memory = sess.memory
        col = self.cfg.qdrant_collection
        t0 = time.monotonic()

        # ----- Step 1: assemble context, tracing every sub-step -----

        context_msgs: list[Message] = []

        if isinstance(memory, BufferMemory):
            # Buffer: return the entire history unchanged.
            all_msgs = await memory._store.get_all(SESSION_ID)
            step(f"[buffer] load all messages from session store → {len(all_msgs)} messages")
            step("[buffer] strategy: no filtering, no ranking — return everything verbatim")
            context_msgs = all_msgs

        elif isinstance(memory, WindowMemory):
            # Window: preserve system msg, trim by count and/or token budget.
            all_msgs = await memory._store.get_all(SESSION_ID)
            step(f"[window] load all messages from session store → {len(all_msgs)} messages")
            has_sys = bool(all_msgs and all_msgs[0].role == "system")
            if has_sys:
                step("[window] system message detected → will be kept regardless of window")
            n_label = f"n={memory._n}" if memory._n else "no count limit"
            b_label = (
                f"token_budget={memory._token_budget}"
                if memory._token_budget
                else "no token budget"
            )
            step(f"[window] applying window ({n_label}, {b_label})")
            context_msgs = await memory.aget_context()
            dropped = len(all_msgs) - len(context_msgs)
            step(f"[window] result: kept {len(context_msgs)} messages, dropped {dropped} oldest")

        elif isinstance(memory, VectorMemory):
            # Vector: embed the query, search Qdrant, return top-k by relevance.
            all_msgs = await memory._store.get_all(SESSION_ID)
            step(f"[vector] session store has {len(all_msgs)} messages, all indexed in Qdrant")
            step(f"[vector] embed query: {text[:70]!r}")
            t_embed = time.monotonic()
            results = await memory.asearch(text, k=memory._top_k)
            elapsed_embed = (time.monotonic() - t_embed) * 1000
            scores_str = ", ".join(f"{r.score:.3f}" for r in results)
            step(f"[vector] Qdrant ANN search [collection={col}, session={SESSION_ID}, "
                 f"top-k={memory._top_k}] [{elapsed_embed:.0f}ms]")
            if results:
                step(f"[vector] retrieved {len(results)} messages — cosine scores: {scores_str}")
            else:
                step("[vector] no results (store may be empty or query has no close match)")
            step("[vector] sort results chronologically for natural reading order")
            context_msgs = sorted((r.message for r in results), key=lambda m: m.timestamp)

        elif isinstance(memory, HybridMemory):
            # Hybrid: semantic retrieval UNION recent window, deduplicated.
            all_msgs = await memory._store.get_all(SESSION_ID)
            step(f"[hybrid] session store has {len(all_msgs)} messages")
            step("[hybrid] ── step 1/3: semantic branch ──────────────────")
            step(f"[hybrid] embed query: {text[:60]!r}")
            t_embed = time.monotonic()
            results = await memory._vector.asearch(text, k=memory._top_k)
            elapsed_embed = (time.monotonic() - t_embed) * 1000
            scores_str = ", ".join(f"{r.score:.3f}" for r in results)
            step(f"[hybrid] Qdrant search [collection={col}, top-k={memory._top_k}]"
                 f" [{elapsed_embed:.0f}ms]")
            step(f"[hybrid] retrieved {len(results)} semantic matches" +
                 (f" (scores: {scores_str})" if scores_str else " (none)"))
            step("[hybrid] ── step 2/3: recency branch ───────────────────")
            step(f"[hybrid] fetch last {memory._window._n} messages from session store")
            window_msgs = await memory._window.aget_context()
            step(f"[hybrid] window: {len(window_msgs)} most recent messages")
            step("[hybrid] ── step 3/3: merge ─────────────────────────────")
            retrieved_ids = {r.message.id for r in results}
            window_ids = {m.id for m in window_msgs}
            overlap = len(retrieved_ids & window_ids)
            new_from_retrieval = len(retrieved_ids - window_ids)
            unique_total = new_from_retrieval + len(window_msgs)
            step(f"[hybrid] deduplicate: {len(results)} semantic + {len(window_msgs)} window "
                 f"= {unique_total} unique ({overlap} already in window, dropped)")
            step("[hybrid] sort merged set chronologically")
            merged: dict[str, Message] = {}
            for msg in window_msgs:
                merged[msg.id] = msg
            for r in results:
                if r.message.id not in window_ids:
                    merged[r.message.id] = r.message
            context_msgs = sorted(merged.values(), key=lambda m: m.timestamp)

        elif isinstance(memory, HierarchicalMemory):
            # Hierarchical: working tail + rolling summary + archival retrieval.
            all_msgs = await memory._store.get_all(SESSION_ID)
            _, body = memory._split_system(all_msgs)
            start = memory._working_start(body)
            working = body[start:]
            archived_count = start - 0  # messages in body before working tail
            working_tokens = count_messages_tokens(working, memory._model)
            sys_msg = all_msgs[0] if (all_msgs and all_msgs[0].role == "system") else None

            step("[hierarchical] ── tier overview ────────────────────────")
            step(f"[hierarchical] total stored : {len(all_msgs)} messages")
            step(f"[hierarchical] working tail : {len(working)} messages "
                 f"({working_tokens} / {memory._working_tokens} token budget)")
            step(f"[hierarchical] archived     : {archived_count} messages indexed in Qdrant")
            step(f"[hierarchical] system msg   : {'present' if sys_msg else 'none'}")
            if memory._summary:
                step(f"[hierarchical] rolling summary: PRESENT "
                     f"(covers first {memory._summarized_count} non-system messages)")
            else:
                step("[hierarchical] rolling summary: none yet (no eviction has occurred)")

            step("[hierarchical] ── archival retrieval ─────────────────────")
            step(f"[hierarchical] embed query: {text[:60]!r}")
            t_arch = time.monotonic()
            arch_results = await memory.asearch(text, k=memory._top_k)
            elapsed_arch = (time.monotonic() - t_arch) * 1000
            working_ids = {m.id for m in working}
            unique_archival = [r for r in arch_results if r.message.id not in working_ids]
            step(f"[hierarchical] Qdrant archival search → {len(arch_results)} hits, "
                 f"{len(unique_archival)} not in working ctx [{elapsed_arch:.0f}ms]")

            step("[hierarchical] ── context assembly ──────────────────────")
            sys_n = 1 if sys_msg else 0
            sum_n = 1 if memory._summary else 0
            total = sys_n + sum_n + len(unique_archival) + len(working)
            step(f"[hierarchical] system({sys_n}) + summary({sum_n}) + "
                 f"archival({len(unique_archival)}) + working({len(working)}) = {total} messages")
            context_msgs = await memory.aget_context(query=text)

        elif isinstance(memory, SummaryMemory):
            # Summary: rolling LLM summary + verbatim recent buffer.
            all_msgs = await memory._store.get_all(SESSION_ID)
            _, body = memory._split_system(all_msgs)
            buffer_msgs = body[memory._summarized_count:]
            summarized = memory._summarized_count
            step("[summary] ── tier overview ────────────────────────")
            step(f"[summary] total stored   : {len(all_msgs)} messages")
            step(f"[summary] summarized     : {summarized} messages folded into rolling summary")
            step(f"[summary] verbatim buffer: {len(buffer_msgs)} recent messages "
                 f"(buffer_size={memory._buffer_size})")
            if memory._summary:
                step(f"[summary] rolling summary: PRESENT ({len(memory._summary)} chars)")
            else:
                step("[summary] rolling summary: none yet (buffer not yet exceeded)")
            context_msgs = await memory.aget_context()
            step(f"[summary] assembled: {len(context_msgs)} messages "
                 f"(system + summary_block + buffer)")

        elif isinstance(memory, FactExtractionMemory):
            # Facts: retrieve relevant facts from Qdrant, return as a system message.
            col = self.cfg.qdrant_collection + "_facts"
            known = len(memory._fact_texts)
            step("[facts] ── fact store overview ───────────────────")
            step(f"[facts] known facts: {known} (in-process list + Qdrant collection={col})")
            step(f"[facts] dedup_threshold: {memory._dedup_threshold}")
            step(f"[facts] embed query: {text[:60]!r}")
            t_embed = time.monotonic()
            results = await memory.asearch(text, k=memory._top_k)
            elapsed_embed = (time.monotonic() - t_embed) * 1000
            if results:
                scores_str = ", ".join(f"{r.score:.3f}" for r in results)
                step(f"[facts] Qdrant facts search → {len(results)} hits "
                     f"(scores: {scores_str}) [{elapsed_embed:.0f}ms]")
            else:
                step(f"[facts] Qdrant facts search → no results yet [{elapsed_embed:.0f}ms]")
            context_msgs = await memory.aget_context(query=text)
            if context_msgs:
                step(f"[facts] context: 1 system message listing {len(results)} relevant facts")
            else:
                step("[facts] context: empty (no facts stored yet)")

        elif isinstance(memory, GraphMemory):
            # Graph: extract entities from query, traverse networkx graph neighbourhood.
            from openmemory.strategies.graph import _extract_entities
            g = memory._graph_store
            nodes = g.node_count(SESSION_ID)
            edges = g.edge_count(SESSION_ID)
            step("[graph] ── graph overview ──────────────────────────")
            step(f"[graph] networkx DiGraph: {nodes} nodes, {edges} edges")
            step(f"[graph] hops={memory._hops} (neighbourhood traversal depth)")
            entities = _extract_entities(text)
            step(f"[graph] extracted entities from query: {entities or '(none detected)'}")
            triplets = await g.neighborhood(SESSION_ID, entities, hops=memory._hops)
            step(f"[graph] neighbourhood traversal → {len(triplets)} relevant triplets")
            context_msgs = await memory.aget_context(query=text)
            if context_msgs:
                step(f"[graph] context: 1 system message with {len(triplets)} triplets")
            else:
                step("[graph] context: empty (no graph built yet)")

        else:
            context_msgs = await sess.aget_context(query=text)

        elapsed_ctx = (time.monotonic() - t0) * 1000
        context = to_openai_messages(context_msgs)
        info(f"context ready: {len(context)} messages [{elapsed_ctx:.0f}ms]")

        # ----- Step 2: call the LLM -----
        messages = context + [{"role": "user", "content": text}]
        step(f"send {len(messages)} messages to {self.cfg.llm_model}...")
        t_llm = time.monotonic()
        try:
            reply = await self.llm.achat(messages)
        except Exception as exc:
            print(f"  [model error: {exc}]")
            return
        elapsed_llm = (time.monotonic() - t_llm) * 1000
        step(f"response received [{elapsed_llm:.0f}ms]")

        print(f"\n{BOLD}[{self.mode}] assistant:{RESET} {reply}\n")

        # ----- Step 3: persist the exchange -----
        new_msgs = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ]
        if self.mode in SEMANTIC:
            em = self.embed_mode
            if em == "paired":
                step(f"embed turn as pair (combined user+assistant text) "
                     f"→ upsert to Qdrant [collection={col}]")
            else:
                step(f"embed 2 messages individually → upsert to Qdrant [collection={col}]")
            step("append 2 messages to session store (full text + metadata)")
        elif self.mode == "summary":
            step("append 2 messages to session store → check if buffer exceeded → "
                 "compress overflow via LLM if needed")
        elif self.mode == "facts":
            step("run fact extraction LLM on user message → dedup → store new facts")
        elif self.mode == "graph":
            step("run triplet extraction LLM on user message → add new edges to graph")
        else:
            step("append 2 messages to session store")

        # Capture pre-aadd state for post-turn reporting.
        pre_evict = getattr(memory, "_summarized_count", None)
        pre_facts = len(memory._fact_texts) if isinstance(memory, FactExtractionMemory) else -1
        pre_graph_edges = (
            memory._graph_store.edge_count(SESSION_ID)
            if isinstance(memory, GraphMemory) else -1
        )
        pre_sum_count = memory._summarized_count if isinstance(memory, SummaryMemory) else -1

        await sess.aadd(new_msgs)

        # Post-aadd result lines.
        post_evict = getattr(memory, "_summarized_count", None)
        if pre_evict is not None and post_evict is not None and post_evict > pre_evict:
            evicted = post_evict - pre_evict
            step(f"[hierarchical] ⚡ working ctx exceeded budget → evicted {evicted} messages")
            step("[hierarchical]    → called LLM summarizer → rolling summary updated")
            step("[hierarchical]    → evicted messages remain searchable via Qdrant")

        if pre_facts >= 0:
            stored = len(memory._fact_texts) - pre_facts
            if stored > 0:
                info(f"[facts] extracted and stored {stored} new fact(s) "
                     f"(total: {len(memory._fact_texts)})")
            else:
                step("[facts] no new facts stored "
                     "(model returned none, or all matched existing facts)")

        if pre_graph_edges >= 0:
            new_edges = memory._graph_store.edge_count(SESSION_ID) - pre_graph_edges
            if new_edges > 0:
                info(f"[graph] added {new_edges} new edge(s) "
                     f"(total: {memory._graph_store.edge_count(SESSION_ID)} edges)")
            else:
                step("[graph] no new triplets extracted")

        if pre_sum_count >= 0:
            folded = memory._summarized_count - pre_sum_count
            if folded > 0:
                step(f"[summary] folded {folded} messages into rolling summary")

        print()

    # -----------------------------------------------------------------------
    # Utility commands
    # -----------------------------------------------------------------------

    async def show_context(self, query: str | None) -> None:
        sess = self.session(self.mode)
        q_label = f"query={query!r}" if query else "no query"
        step(f"assembling context for mode='{self.mode}' ({q_label})")
        ctx = await sess.aget_context(query=query)
        info(f"context in '{self.mode}' mode: {len(ctx)} messages")
        for m in to_openai_messages(ctx):
            preview = m["content"][:100].replace("\n", " ")
            tag = f"  {m['role']}: "
            print(f"{DIM}{tag}{RESET}{preview}")

    async def search(self, query: str) -> None:
        if self.mode not in SEMANTIC:
            warn(f"'{self.mode}' has no vector index — switch to vector/hybrid/hierarchical first")
            return
        sess = self.session(self.mode)
        col = self.cfg.qdrant_collection
        step(f"embed query: {query!r}")
        step(f"search Qdrant [collection={col}, session={SESSION_ID}, top-k=5]")
        results = await sess.asearch(query, k=5)
        if not results:
            print("  (no results)")
            return
        info(f"{len(results)} results:")
        for r in results:
            print(f"  score={r.score:.3f}  {DIM}{r.message.content[:90]}{RESET}")

    async def history(self) -> None:
        msgs = await self.mem._session_store().get_all(SESSION_ID)
        info(f"full session history: {len(msgs)} messages")
        for m in to_openai_messages(msgs):
            print(f"  {DIM}{m['role']}:{RESET} {m['content'][:100]}")

    async def clear(self) -> None:
        step("deleting messages from session store...")
        if self.mode in SEMANTIC:
            step("removing vectors from Qdrant for this session...")
        elif self.mode == "facts":
            step("removing fact vectors from Qdrant facts collection...")
        await self.session(self.mode).aclear()
        # Reset in-process state on cached strategy instances.
        for _key, sess_obj in self._sessions.items():
            mem = sess_obj.memory  # type: ignore[attr-defined]
            if isinstance(mem, (HierarchicalMemory, SummaryMemory)):
                mem._summary = ""
                mem._summarized_count = 0
            elif isinstance(mem, FactExtractionMemory):
                mem._fact_texts.clear()
        print("  session cleared")

    async def run(self) -> None:
        _setup_readline()
        banner(self.mode, PROVIDER)
        print(f"{DIM}  internal steps shown in grey  —  /help for commands{RESET}\n")
        while True:
            em_label = (
                f",embed={self.embed_mode}" if self.mode in SEMANTIC else ""
            )
            prompt = f"({self.mode}{em_label}) you> "
            try:
                line = (await asyncio.to_thread(input, prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, arg = line[1:].partition(" ")
                arg = arg.strip()
                if cmd in {"quit", "exit"}:
                    break
                elif cmd == "help":
                    print(HELP)
                elif cmd == "mode":
                    await self.switch(arg)
                elif cmd in {"embed-mode", "embed_mode"}:
                    await self.switch_embed_mode(arg)
                elif cmd == "context":
                    await self.show_context(arg or None)
                elif cmd == "search":
                    await self.search(arg)
                elif cmd == "history":
                    await self.history()
                elif cmd == "clear":
                    await self.clear()
                else:
                    print(f"  unknown command /{cmd} — try /help")
                continue
            await self.say(line)

        await self.mem.aclose()
        print("bye!")


if __name__ == "__main__":
    asyncio.run(LiveChat().run())
