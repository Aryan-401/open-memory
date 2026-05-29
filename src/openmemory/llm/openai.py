"""OpenAI-compatible chat LLM (core)."""

from __future__ import annotations

from typing import Any

from ..config import Config
from .base import LLM


class OpenAILLM(LLM):
    def __init__(self, config: Config) -> None:
        from openai import AsyncOpenAI

        self._model = config.llm_model
        self._client = AsyncOpenAI(
            api_key=config.openai_api_key, base_url=config.openai_base_url
        )

    async def achat(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.2
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
