"""LiteLLM chat LLM ([litellm] extra)."""

from __future__ import annotations

from typing import Any

from ..config import Config
from .base import LLM


class LiteLLMLLM(LLM):
    def __init__(self, config: Config) -> None:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LiteLLM provider requires the 'litellm' extra. "
                "Install with: pip install 'open-memory[litellm]'"
            ) from exc
        self._litellm = litellm
        self._model = config.llm_model

    async def achat(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.2
    ) -> str:
        resp = await self._litellm.acompletion(
            model=self._model, messages=messages, temperature=temperature
        )
        return resp["choices"][0]["message"]["content"] or ""
