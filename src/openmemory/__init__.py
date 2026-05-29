"""open-memory: pluggable context-management strategies for LLM applications.

Quick start::

    from openmemory import OpenMemory

    mem = OpenMemory()                       # in-memory, buffer by default
    chat = mem.session("user-42", strategy="hybrid")
    await chat.aadd({"role": "user", "content": "I love hiking in the Alps"})
    messages = await chat.aget_openai_context(query="vacation ideas")
    # -> list[{"role", "content"}] ready to pass to an OpenAI-compatible API
"""

from __future__ import annotations

from .client import OpenMemory, Strategy
from .config import Config
from .core.base import BaseMemory
from .core.models import EmbedMode, Message, RetrievalResult, embed_texts, to_openai_messages
from .session import Session
from .strategies.buffer import BufferMemory
from .strategies.facts import FactExtractionMemory
from .strategies.graph import GraphMemory
from .strategies.hierarchical import HierarchicalMemory
from .strategies.hybrid import HybridMemory
from .strategies.summary import SummaryMemory
from .strategies.vector import VectorMemory
from .strategies.window import WindowMemory

__version__ = "0.1.0"

__all__ = [
    "OpenMemory",
    "Strategy",
    "Config",
    "Session",
    "EmbedMode",
    "Message",
    "RetrievalResult",
    "embed_texts",
    "to_openai_messages",
    "BaseMemory",
    "BufferMemory",
    "WindowMemory",
    "VectorMemory",
    "HybridMemory",
    "HierarchicalMemory",
    "SummaryMemory",
    "FactExtractionMemory",
    "GraphMemory",
]
