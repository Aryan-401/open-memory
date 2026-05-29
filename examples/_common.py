"""Shared provider wiring for the examples.

Pick a provider and supply a key in one of two ways:

  1. Environment variables (recommended)::

         export OPENMEMORY_EXAMPLE_PROVIDER=openai   # claude|gemini|groq|huggingface|litellm
         export OPENAI_API_KEY=sk-...                # the provider's own key var

  2. Or paste directly below into ``API_KEY`` (and set ``PROVIDER``).

Chat for Claude / Gemini / Groq is routed through LiteLLM (one auth surface for 100+
providers); OpenAI is used natively. ``huggingface`` runs a small model locally with no
key. Embeddings default to a local sentence-transformers model where the provider has no
embeddings API (Claude, Groq), so the vector examples work without a second key.

Install the deps with::

    pip install "open-memory[examples]"
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openmemory import Config

# --- paste your key here if you prefer not to use env vars ---
PROVIDER = os.getenv("OPENMEMORY_EXAMPLE_PROVIDER", "openai")
API_KEY = os.getenv("OPENMEMORY_EXAMPLE_API_KEY", "")  # e.g. API_KEY = "sk-..."


@dataclass(frozen=True)
class Preset:
    llm_provider: str
    llm_model: str
    embedder_provider: str
    embedding_model: str | None
    key_env: str | None  # the provider's own API-key environment variable


PRESETS: dict[str, Preset] = {
    "openai": Preset(
        "openai", "gpt-4o-mini", "openai", "text-embedding-3-small", "OPENAI_API_KEY"
    ),
    "claude": Preset(
        "litellm", "anthropic/claude-3-5-haiku-latest", "local", None, "ANTHROPIC_API_KEY"
    ),
    "gemini": Preset(
        "litellm", "gemini/gemini-1.5-flash", "litellm", "gemini/text-embedding-004",
        "GEMINI_API_KEY",
    ),
    "groq": Preset(
        "litellm", "groq/llama-3.1-8b-instant", "local", None, "GROQ_API_KEY"
    ),
    # Fully local — downloads a small model on first run, no API key required.
    "huggingface": Preset(
        "local", "HuggingFaceTB/SmolLM2-360M-Instruct", "local", None, None
    ),
    # Generic LiteLLM passthrough: set OPENMEMORY_LLM_MODEL + the matching key var.
    "litellm": Preset(
        "litellm", os.getenv("OPENMEMORY_LLM_MODEL", "gpt-4o-mini"), "local", None, None
    ),
}


def build_config(provider: str = PROVIDER, api_key: str = API_KEY) -> Config:
    """Construct an open-memory ``Config`` for the selected provider."""
    if provider not in PRESETS:
        raise SystemExit(
            f"Unknown provider {provider!r}. Choose one of: {', '.join(PRESETS)}"
        )
    preset = PRESETS[provider]

    kwargs: dict[str, object] = {
        "llm_provider": preset.llm_provider,
        "llm_model": preset.llm_model,
        "embedder_provider": preset.embedder_provider,
    }
    if preset.embedding_model:
        kwargs["embedding_model"] = preset.embedding_model

    cfg = Config(**kwargs)  # type: ignore[arg-type]

    key = api_key or (os.getenv(preset.key_env) if preset.key_env else None)
    if key and preset.key_env:
        if provider == "openai":
            cfg.openai_api_key = key  # used by both the OpenAI LLM and embedder
        else:
            os.environ[preset.key_env] = key  # LiteLLM reads keys from the environment
    return cfg


def banner(strategy: str, provider: str = PROVIDER) -> None:
    """Print what we're about to run and warn if a required key is missing."""
    preset = PRESETS[provider]
    print(f"▶ strategy={strategy}  provider={provider}  model={preset.llm_model}")
    if preset.key_env:
        has_key = bool(API_KEY) or bool(os.getenv(preset.key_env))
        if not has_key:
            print(
                f"  ⚠ No API key found. Set {preset.key_env} (or paste into _common.API_KEY) "
                f"before running."
            )
