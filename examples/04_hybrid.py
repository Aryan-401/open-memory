"""HybridMemory — recent window UNION semantic retrieval (the recommended default).

The model always sees the latest turns *and* older-but-relevant turns surfaced by the
query. Here an important early fact resurfaces even after lots of unrelated chatter.

    python examples/04_hybrid.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("hybrid")
    cfg = build_config()
    mem = OpenMemory(cfg)
    chat = mem.session("demo-hybrid", strategy="hybrid", window_size=4, top_k=3)

    # An important fact, stated early...
    await chat.aadd({"role": "user", "content": "Important: my flight is on June 14th."})
    # ...buried under unrelated chatter.
    for topic in ["the weather", "a movie", "lunch", "my new headphones", "weekend plans"]:
        await chat.aadd({"role": "user", "content": f"Let's chat about {topic}."})
        await chat.aadd({"role": "assistant", "content": f"Sure, {topic} sounds fun!"})

    question = "When is my flight again?"
    context = await chat.aget_openai_context(query=question)
    print(f"\n[hybrid context: {len(context)} messages = recent window + retrieved]")
    for m in context:
        print(f"  {m['role']}: {m['content']}")

    llm = build_llm(cfg)
    reply = await llm.achat(context + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")
    await mem.aclose()


if __name__ == "__main__":
    asyncio.run(main())
