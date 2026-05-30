"""FactExtractionMemory — atomic fact extraction and deduplication (mem0-style).

Instead of storing raw turns, an LLM distils each exchange into a list of durable atomic
facts ("user is vegetarian", "deadline is Friday"). New facts are deduplicated against
existing ones via cosine similarity before being stored in a dedicated Qdrant collection.
``aget_context`` retrieves the facts most relevant to the current query (or all known
facts when no query is given) and returns them as a single system message.

LLM calls: one extraction call per ``aadd`` (over the new messages) plus one embedding
call per candidate fact (for dedup). No second LLM call for reconciliation — similarity
threshold handles deduplication.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.base import BaseMemory
from ..core.models import Message, RetrievalResult
from ..embeddings.base import Embedder
from ..llm.base import LLM
from ..storage.qdrant_store import QdrantVectorStore
from ..storage.session_store import SessionStore

_JSON_ARRAY = re.compile(r'\[.*\]', re.DOTALL)

_EXTRACTION_SYSTEM = """\
Extract facts about the user from the conversation turns below.
Return ONLY a valid JSON array of short fact strings. No other text.

Include anything the user says about themselves or their situation:
preferences, opinions, names, jobs, dates, plans, locations, constraints, \
or casual mentions like "I like X" or "I'm a Y".

Do NOT invent or infer facts that were not stated.

Examples:
  user: "My name is Alice, I work at Google."
  → ["name is Alice", "works at Google"]

  user: "I like pie. Especially apple pie."
  → ["likes pie", "likes apple pie"]

  user: "Remind me about the meeting on Friday."
  → ["has a meeting on Friday"]

  user: "Thanks!"
  → []

Return [] if the user said nothing factual about themselves."""


def _parse_json_list(text: str) -> list[Any]:
    """Extract the outermost JSON array from a possibly noisy LLM response."""
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


class FactExtractionMemory(BaseMemory):
    """mem0-style memory: extract → deduplicate → store atomic facts.

    Facts are stored in a dedicated Qdrant collection (separate from the main message
    store) so they can be retrieved by semantic similarity without mixing with raw turns.
    Raw messages are also appended to the session store for auditing and ``/history``.

    Parameters
    ----------
    top_k:
        Maximum number of facts returned by ``aget_context``.
    dedup_threshold:
        Cosine similarity above which an incoming fact is considered a duplicate of an
        existing one and is silently skipped.
    """

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        llm: LLM,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        *,
        top_k: int = 10,
        dedup_threshold: float = 0.85,
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._llm = llm
        self._embedder = embedder
        self._vectors = vector_store
        self._top_k = top_k
        self._dedup_threshold = dedup_threshold
        # In-process list for query-free recall (rebuilt from Qdrant is a future enhancement).
        self._fact_texts: list[str] = []

    async def aadd(
        self, messages: Message | dict[str, Any] | list[Message | dict[str, Any]]
    ) -> None:
        msgs = self._coerce(messages)
        if not msgs:
            return
        await self._store.append(self.session_id, msgs)
        # Only extract from turns that contain user content — assistant-only batches
        # (e.g. when messages are added one at a time) have no facts worth extracting
        # and the extra LLM call just burns API quota.
        if any(m.role == "user" for m in msgs):
            facts = await self._extract_facts(msgs)
            if facts:
                await self._reconcile_and_store(facts)

    async def _extract_facts(self, msgs: list[Message]) -> list[str]:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in msgs)
        try:
            response = await self._llm.achat(
                [
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": transcript},
                ],
                temperature=0,  # deterministic output for structured JSON
            )
            parsed = _parse_json_list(response)
            return [f for f in parsed if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            return []
        except Exception as exc:
            import sys
            print(f"  [open-memory] fact extraction failed: {exc}", file=sys.stderr)
            return []

    async def _reconcile_and_store(self, facts: list[str]) -> None:
        for fact in facts:
            vec = await self._embedder.aembed_one(fact)
            similar = await self._vectors.search(self.session_id, vec, k=1)
            if similar and similar[0].score >= self._dedup_threshold:
                continue  # already known — skip
            fact_msg = Message(
                role="system",
                content=fact,
                session_id=self.session_id,
                metadata={"openmemory_kind": "fact"},
            )
            await self._vectors.upsert([fact_msg], [vec])
            self._fact_texts.append(fact)

    async def aget_context(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        k = limit or self._top_k
        if query is not None:
            results = await self.asearch(query, k=k)
            facts = [r.message.content for r in results]
        else:
            facts = self._fact_texts[:k]

        if not facts:
            return []
        return [
            Message(
                role="system",
                content="Known facts:\n" + "\n".join(f"- {f}" for f in facts),
                session_id=self.session_id,
                metadata={"openmemory_kind": "facts_context"},
            )
        ]

    async def asearch(self, query: str, k: int = 5) -> list[RetrievalResult]:
        vec = await self._embedder.aembed_one(query)
        return await self._vectors.search(self.session_id, vec, k=k)

    async def aclear(self) -> None:
        await self._store.clear(self.session_id)
        await self._vectors.clear(self.session_id)
        self._fact_texts.clear()
