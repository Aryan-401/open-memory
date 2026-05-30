"""FactExtractionMemory — atomic fact extraction and deduplication (mem0-style).

Each turn is processed by an LLM that extracts durable facts ("user is vegetarian",
"deadline is Friday"). New facts are checked against existing ones via cosine similarity;
duplicates are silently skipped. Context is the most relevant facts for the current query.

    python examples/08_facts.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("facts")
    cfg = build_config()
    mem = OpenMemory(cfg)
    chat = mem.session("demo-facts", strategy="facts")

    print("\n--- adding turns (facts extracted after each) ---")
    turns = [
        ("user", "My name is Jordan and I work as a data scientist at Acme Corp."),
        ("assistant", "Nice to meet you, Jordan!"),
        ("user", "I use Python and SQL daily. I'm learning Rust in my spare time."),
        ("assistant", "Great stack! Rust is fantastic for systems work."),
        ("user", "My team's project deadline is August 15th. We're building a fraud detector."),
        ("assistant", "That's an important project! Fraud detection needs high precision."),
        ("user", "I prefer async code over sync and I always write unit tests."),
        ("assistant", "Good practices! Async Python with pytest-asyncio is a solid combo."),
        # Intentional near-duplicate — should be deduplicated.
        ("user", "By the way, I'm a data scientist working with Python every day."),
        ("assistant", "You mentioned that — great to have a Python expert on the team!"),
    ]
    for role, content in turns:
        await chat.aadd({"role": role, "content": content})
        if role == "assistant":
            # Peek at how many facts are stored after each exchange.
            from openmemory.strategies.facts import FactExtractionMemory
            mem_obj = chat.memory
            if isinstance(mem_obj, FactExtractionMemory):
                print(f"  [facts stored so far: {len(mem_obj._fact_texts)}]")

    print("\n--- all known facts (no query) ---")
    ctx = await chat.aget_openai_context()
    for m in ctx:
        print(f"  {m['content']}")

    print("\n--- query-focused facts ---")
    ctx_q = await chat.aget_openai_context(query="programming tools and deadlines")
    for m in ctx_q:
        print(f"  {m['content']}")

    question = "What do you know about my work and preferences?"
    llm = build_llm(cfg)
    reply = await llm.achat(ctx_q + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")
    await mem.aclose()


if __name__ == "__main__":
    asyncio.run(main())
