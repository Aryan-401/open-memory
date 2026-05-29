"""Local HuggingFace chat LLM ([local-llm] extra).

A pragmatic offline summarization LLM built on ``transformers`` pipelines. Generation is
synchronous, so it is dispatched to a worker thread. Intended for summarization-style
workloads (used by :class:`~openmemory.llm.summarizer.Summarizer`), not high-throughput
serving.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..config import Config
from .base import LLM


class LocalHFLLM(LLM):
    def __init__(self, config: Config) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Local LLM requires the 'local-llm' extra. "
                "Install with: pip install 'open-memory[local-llm]'"
            ) from exc
        self._pipe = pipeline("text-generation", model=config.llm_model)

    @staticmethod
    def _to_prompt(messages: list[dict[str, Any]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    async def achat(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.2
    ) -> str:
        prompt = self._to_prompt(messages)
        out = await asyncio.to_thread(
            self._pipe,
            prompt,
            max_new_tokens=512,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()
