"""BufferMemory — replay the entire conversation verbatim.

The simplest strategy: every turn is kept and fed back. Good for short chats.

    pip install "open-memory[examples]"
    export OPENMEMORY_EXAMPLE_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/01_buffer.py
"""

from __future__ import annotations

import asyncio

from _common import banner, build_config

from openmemory import OpenMemory
from openmemory.llm.base import build_llm


async def main() -> None:
    banner("buffer")
    cfg = build_config()
    mem = OpenMemory(cfg)
    chat = mem.session("demo-buffer", strategy="buffer")

    await chat.aadd(
        [
            {"role": "system", "content": "You are a concise travel assistant."},
            {"role": "user", "content": "Hi! I'm planning a trip."},
            {"role": "assistant", "content": "Wonderful — what kind of place appeals to you?"},
            {"role": "user", "content": "Mountains, great food, and not too touristy."},
        ]
    )

    question = "Based on everything so far, suggest exactly one destination and why."
    context = await chat.aget_openai_context()
    print(f"\n[context has {len(context)} messages — the full history]")

    llm = build_llm(cfg)
    reply = await llm.achat(context + [{"role": "user", "content": question}])
    print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
