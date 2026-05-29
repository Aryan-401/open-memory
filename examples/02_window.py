"""WindowMemory — keep only the most recent turns (plus the system prompt).

We add many turns but window to the last few; the system message is always retained.

    python examples/02_window.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("window")
    cfg = build_config()
    mem = OpenMemory(cfg)
    # Keep only the last 4 turns.
    chat = mem.session("demo-window", strategy="window", n=4)

    await chat.aadd({"role": "system", "content": "You are a helpful assistant."})
    for i in range(1, 9):
        await chat.aadd({"role": "user", "content": f"Fact #{i}: my lucky number is {i}."})
        await chat.aadd({"role": "assistant", "content": f"Noted fact #{i}."})

    question = "What is the most recent fact I told you?"
    context = await chat.aget_openai_context()
    print(f"\n[context windowed to {len(context)} messages — system + last 4 turns]")
    for m in context:
        print(f"  {m['role']}: {m['content']}")

    llm = build_llm(cfg)
    reply = await llm.achat(context + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
