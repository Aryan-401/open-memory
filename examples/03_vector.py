"""VectorMemory — semantic retrieval over Qdrant.

We store a pile of unrelated facts, then ask a question; only the semantically relevant
fact is pulled back into context.

Needs embeddings. With provider=openai/gemini that's an API call; with claude/groq/
huggingface it falls back to a local sentence-transformers model (downloaded on first run).
An in-memory Qdrant is used unless OPENMEMORY_QDRANT_URL is set.

    python examples/03_vector.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("vector")
    cfg = build_config()
    mem = OpenMemory(cfg)
    chat = mem.session("demo-vector", strategy="vector", top_k=2)

    await chat.aadd(
        [
            {"role": "user", "content": "I'm allergic to peanuts."},
            {"role": "user", "content": "My favorite programming language is Rust."},
            {"role": "user", "content": "I have a golden retriever named Biscuit."},
            {"role": "user", "content": "I prefer window seats on flights."},
        ]
    )

    question = "Is there anything I should avoid in my food?"
    results = await chat.asearch(question, k=2)
    print("\n[top semantic matches]")
    for r in results:
        print(f"  score={r.score:.3f}  {r.message.content}")

    context = await chat.aget_openai_context(query=question)
    llm = build_llm(cfg)
    reply = await llm.achat(context + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")
    await mem.aclose()


if __name__ == "__main__":
    asyncio.run(main())
