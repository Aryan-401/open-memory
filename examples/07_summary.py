"""SummaryMemory — rolling summary + verbatim recent buffer.

Adds 10 turns. With a small buffer (4 messages) the older turns are folded into a rolling
summary by the LLM. The assembled context is: [summary-system-msg] + [4 recent messages].
This keeps context bounded while nothing is truly lost.

    python examples/07_summary.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("summary")
    cfg = build_config()
    mem = OpenMemory(cfg)
    # Small buffer forces compression quickly for the demo.
    chat = mem.session("demo-summary", strategy="summary", buffer_size=4)

    print("\n--- adding 10 turns ---")
    facts = [
        ("user", "My name is Alex and I'm planning a trip to Japan."),
        ("assistant", "That sounds wonderful! Japan has so much to offer."),
        ("user", "I want to visit Kyoto and Tokyo. My budget is $3000."),
        ("assistant", "Great choice! Kyoto has temples; Tokyo is modern and vibrant."),
        ("user", "I prefer vegetarian food and I don't drink alcohol."),
        ("assistant", "Noted — Japan has great shojin ryori (Buddhist vegetarian cuisine)."),
        ("user", "My flight departs on July 12th."),
        ("assistant", "Got it. July is warm and humid in Japan — pack light clothes."),
        ("user", "I'll be travelling solo for 14 days."),
        ("assistant", "Solo travel in Japan is very safe and easy to navigate."),
    ]
    for role, content in facts:
        await chat.aadd({"role": role, "content": content})

    print("\n--- assembled context ---")
    ctx = await chat.aget_openai_context()
    for m in ctx:
        kind = m.get("metadata", {}).get("openmemory_kind", "")  # type: ignore[call-overload]
        label = f"[{kind}] " if kind else ""
        print(f"  {m['role']}: {label}{m['content'][:100]}")

    question = "Remind me of my trip details."
    llm = build_llm(cfg)
    reply = await llm.achat(ctx + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")
    await mem.aclose()


if __name__ == "__main__":
    asyncio.run(main())
