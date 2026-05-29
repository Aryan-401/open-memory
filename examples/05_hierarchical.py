"""HierarchicalMemory — MemGPT-style working context + rolling summary + archival.

We use a tiny working-context budget so older turns get evicted: they are folded into a
rolling summary (via the provider's LLM) and paged to archival storage, where they remain
retrievable by query. The assembled context is: system + summary + retrieved + recent.

Needs both an LLM (for summarization) and embeddings.

    python examples/05_hierarchical.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("hierarchical")
    cfg = build_config()
    mem = OpenMemory(cfg)
    # Tiny budget forces eviction + summarization quickly for the demo.
    chat = mem.session(
        "demo-hierarchical", strategy="hierarchical", working_context_tokens=120, top_k=3
    )

    await chat.aadd({"role": "user", "content": "Remember: my hotel confirmation code is ZX-9981."})
    for i in range(1, 11):
        await chat.aadd({"role": "user", "content": f"Day {i}: tell me a fun travel fact."})
        await chat.aadd(
            {"role": "assistant", "content": f"Fun fact {i}: travel broadens the mind!"}
        )

    question = "What's my hotel confirmation code?"
    context = await chat.aget_openai_context(query=question)
    print(f"\n[hierarchical context: {len(context)} messages]")
    for m in context:
        preview = m["content"][:80].replace("\n", " ")
        print(f"  {m['role']}: {preview}")

    llm = build_llm(cfg)
    reply = await llm.achat(context + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")
    await mem.aclose()


if __name__ == "__main__":
    asyncio.run(main())
