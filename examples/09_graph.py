"""GraphMemory — knowledge graph via networkx (entities + relationships).

Each turn is processed by an LLM that extracts (subject, predicate, object) triplets.
These accumulate in a session-scoped networkx DiGraph. ``aget_context(query=...)``
traverses the graph neighbourhood around entities in the query and returns the relevant
subgraph as a system message.

    python examples/09_graph.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm
from openmemory.strategies.graph import GraphMemory


async def main() -> None:
    banner("graph")
    cfg = build_config()
    mem = OpenMemory(cfg)
    chat = mem.session("demo-graph", strategy="graph", hops=2)

    print("\n--- adding turns (triplets extracted after each) ---")
    turns = [
        ("user", "Alice is the project lead for Project Atlas. She works with Bob and Carol."),
        ("assistant", "Got it — Alice leads Atlas with Bob and Carol."),
        ("user", "Bob owns the backend API. Carol is responsible for the frontend dashboard."),
        ("assistant", "Clear ownership: Bob → backend, Carol → frontend."),
        ("user", "Project Atlas deadline is September 30th. It's a high-priority initiative."),
        ("assistant", "Noted — Atlas is high-priority, deadline September 30th."),
        ("user", "Alice reports to Dan, the VP of Engineering."),
        ("assistant", "So Dan is the VP and Alice's manager."),
    ]
    for role, content in turns:
        await chat.aadd({"role": role, "content": content})
        if role == "assistant":
            mem_obj = chat.memory
            if isinstance(mem_obj, GraphMemory):
                g = mem_obj._graph_store
                print(
                    f"  [graph: {g.node_count('demo-graph')} nodes, "
                    f"{g.edge_count('demo-graph')} edges]"
                )

    print("\n--- full graph (no query) ---")
    ctx_all = await chat.aget_openai_context()
    for m in ctx_all:
        print(f"  {m['content']}")

    print("\n--- graph neighbourhood for 'Alice' query ---")
    ctx_q = await chat.aget_openai_context(query="Who does Alice work with?")
    for m in ctx_q:
        print(f"  {m['content']}")

    question = "Give me a status update on Project Atlas."
    llm = build_llm(cfg)
    reply = await llm.achat(ctx_q + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")
    await mem.aclose()


if __name__ == "__main__":
    asyncio.run(main())
