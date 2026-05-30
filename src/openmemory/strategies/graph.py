"""GraphMemory — knowledge graph memory (networkx backend, Neo4j/Graphiti planned).

An LLM extracts (subject, predicate, object) triplets from each conversation turn and
accumulates them into a session-scoped ``networkx.DiGraph``. ``aget_context(query=...)``
extracts entity names from the query, traverses the graph neighbourhood around those
entities (up to ``hops`` hops), and returns the relevant subgraph as a system message.

This captures relational structure ("who works with whom", "which task blocks which")
that flat vector retrieval misses.

Future backend: swap ``NetworkxGraphStore`` for a ``Neo4jGraphStore`` / ``GraphitiStore``
that implements the same ``GraphStore`` protocol — zero changes to ``GraphMemory`` itself.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult
from ..llm.base import LLM
from ..storage.session_store import SessionStore

_JSON_ARRAY = re.compile(r'\[.*\]', re.DOTALL)

_TRIPLET_SYSTEM = """\
Extract knowledge graph triplets from the conversation turns below.
Return ONLY a valid JSON array of [subject, predicate, object] arrays.
Rules:
- Use concise, lowercase, underscore-joined predicates: works_with, deadline_is, \
located_in, prefers, is_a, has_role, owns, part_of, etc.
- Normalize entity names consistently (always the same string for the same person/thing).
- Extract facts stated or strongly implied by the user, not the assistant's filler.

Example output:
[["Alice", "works_with", "Bob"], ["project_x", "deadline_is", "Friday"], \
["user", "prefers", "dark_mode"]]

If no triplets can be extracted, return an empty array: []"""


def _parse_json_list(text: str) -> list[Any]:
    """Extract the outermost JSON array from a possibly noisy LLM response."""
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def _extract_entities(text: str) -> list[str]:
    """Heuristic: capitalised words and multi-word proper nouns as candidate entities."""
    words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
    return list(dict.fromkeys(words))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# GraphStore protocol (stable interface for future backends)
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphStore(Protocol):
    """Backend contract for graph-based memory (networkx, Neo4j, Graphiti, ...)."""

    async def add_triplets(
        self, session_id: str, triplets: list[tuple[str, str, str]]
    ) -> None: ...

    async def neighborhood(
        self, session_id: str, entities: list[str], hops: int = 1
    ) -> list[tuple[str, str, str]]: ...

    def all_triplets(self, session_id: str) -> list[tuple[str, str, str]]: ...

    async def clear(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# NetworkxGraphStore — in-process backend
# ---------------------------------------------------------------------------

class NetworkxGraphStore:
    """In-process networkx DiGraph, one per session_id.

    No persistence across restarts (v1). A future SQLite/Neo4j backend can implement
    the ``GraphStore`` protocol and be swapped in without touching ``GraphMemory``.
    """

    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}

    def _graph(self, session_id: str) -> Any:
        if session_id not in self._graphs:
            import networkx as nx  # core dep — always present
            self._graphs[session_id] = nx.DiGraph()
        return self._graphs[session_id]

    async def add_triplets(
        self, session_id: str, triplets: list[tuple[str, str, str]]
    ) -> None:
        g = self._graph(session_id)
        for subj, pred, obj in triplets:
            # If the edge already exists, keep the existing predicate (first-write wins).
            if not g.has_edge(subj, obj):
                g.add_edge(subj, obj, predicate=pred)

    async def neighborhood(
        self, session_id: str, entities: list[str], hops: int = 1
    ) -> list[tuple[str, str, str]]:
        g = self._graph(session_id)
        if not g.nodes:
            return []

        subgraph_nodes: set[str] = set()
        for entity in entities:
            # Case-insensitive match against node names.
            matched = [n for n in g.nodes if entity.lower() in n.lower()]
            for node in matched:
                subgraph_nodes.add(node)
                frontier = {node}
                for _ in range(hops):
                    next_frontier: set[str] = set()
                    for n in frontier:
                        next_frontier.update(g.successors(n))
                        next_frontier.update(g.predecessors(n))
                    subgraph_nodes.update(next_frontier)
                    frontier = next_frontier

        return [
            (u, data.get("predicate", "—"), v)
            for u, v, data in g.edges(data=True)
            if u in subgraph_nodes or v in subgraph_nodes
        ]

    def all_triplets(self, session_id: str) -> list[tuple[str, str, str]]:
        g = self._graphs.get(session_id)
        if g is None:
            return []
        return [
            (u, data.get("predicate", "—"), v)
            for u, v, data in g.edges(data=True)
        ]

    def node_count(self, session_id: str) -> int:
        g = self._graphs.get(session_id)
        return len(g.nodes) if g is not None else 0

    def edge_count(self, session_id: str) -> int:
        g = self._graphs.get(session_id)
        return len(g.edges) if g is not None else 0

    async def clear(self, session_id: str) -> None:
        self._graphs.pop(session_id, None)


# ---------------------------------------------------------------------------
# GraphMemory strategy
# ---------------------------------------------------------------------------

class GraphMemory(BaseMemory):
    """Knowledge graph memory: entities and relationships extracted from conversation.

    Parameters
    ----------
    hops:
        Neighbourhood traversal depth when querying the graph. 1 = direct neighbours.
    """

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        llm: LLM,
        graph_store: NetworkxGraphStore,
        *,
        hops: int = 1,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._llm = llm
        self._graph_store = graph_store
        self._hops = hops

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        msgs = self._coerce(messages)
        if not msgs:
            return
        await self._store.append(self.session_id, msgs)
        # Only extract from turns that contain user content — assistant-only batches
        # rarely introduce new entities worth adding to the graph.
        if any(m.role == "user" for m in msgs):
            triplets = await self._extract_triplets(msgs)
            if triplets:
                await self._graph_store.add_triplets(self.session_id, triplets)

    async def _extract_triplets(
        self, msgs: list[Message]
    ) -> list[tuple[str, str, str]]:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in msgs)
        try:
            response = await self._llm.achat(
                [
                    {"role": "system", "content": _TRIPLET_SYSTEM},
                    {"role": "user", "content": transcript},
                ],
                temperature=0,  # deterministic output for structured JSON
            )
            parsed = _parse_json_list(response)
            return [
                (str(t[0]), str(t[1]), str(t[2]))
                for t in parsed
                if isinstance(t, list) and len(t) == 3
            ]
        except json.JSONDecodeError:
            return []
        except Exception as exc:
            import sys
            print(f"  [open-memory] triplet extraction failed: {exc}", file=sys.stderr)
            return []

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        if query is not None:
            entities = _extract_entities(query)
            triplets = await self._graph_store.neighborhood(
                self.session_id, entities, hops=self._hops
            )
        else:
            triplets = self._graph_store.all_triplets(self.session_id)

        if not triplets:
            return []

        lines = "\n".join(f"- {s} {p} {o}" for s, p, o in triplets)
        return [
            Message(
                role="system",
                content=f"Knowledge graph context:\n{lines}",
                session_id=self.session_id,
                metadata={"openmemory_kind": "graph_context"},
            )
        ]

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        # Graph memory has no vector index — use aget_context for graph traversal.
        return []

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
        await self._graph_store.clear(self.session_id)
