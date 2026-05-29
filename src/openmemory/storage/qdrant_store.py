"""Async Qdrant-backed vector store for semantic retrieval.

Uses a single collection with a ``session_id`` payload filter (rather than a collection
per session) — this scales to many sessions and keeps index management trivial. The full
:class:`Message` is stored in the point payload so search returns complete records.

When ``url`` is ``None`` a local in-memory Qdrant is used (``location=":memory:"``),
which is ideal for tests and quick experiments with no Docker required.
"""

from __future__ import annotations

import asyncio
import warnings

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from ..core.models import Message, RetrievalResult


class QdrantVectorStore:
    """Thin async wrapper around a Qdrant collection."""

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "openmemory",
    ) -> None:
        if url:
            self._client = AsyncQdrantClient(url=url, api_key=api_key)
        else:
            self._client = AsyncQdrantClient(location=":memory:")
        self._collection = collection
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure_collection(self, dim: int) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=dim, distance=qmodels.Distance.COSINE
                    ),
                )
                # Index session_id so filtered queries stay fast at scale.
                # Local (in-memory) mode ignores payload indexes (and warns); suppress
                # the warning there and tolerate backends that reject the call.
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        await self._client.create_payload_index(
                            collection_name=self._collection,
                            field_name="session_id",
                            field_schema=qmodels.PayloadSchemaType.KEYWORD,
                        )
                except (ValueError, RuntimeError):
                    pass
            self._ready = True

    async def upsert(self, messages: list[Message], vectors: list[list[float]]) -> None:
        if not messages:
            return
        await self._ensure_collection(len(vectors[0]))
        points = [
            qmodels.PointStruct(
                id=msg.id,
                vector=vec,
                payload={"session_id": msg.session_id, "message": msg.model_dump(mode="json")},
            )
            for msg, vec in zip(messages, vectors, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self, session_id: str, vector: list[float], k: int = 5
    ) -> list[RetrievalResult]:
        await self._ensure_collection(len(vector))
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="session_id", match=qmodels.MatchValue(value=session_id)
                )
            ]
        )
        hits = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=flt,
            limit=k,
            with_payload=True,
        )
        results: list[RetrievalResult] = []
        for point in hits.points:
            payload = point.payload or {}
            msg = Message.model_validate(payload["message"])
            results.append(RetrievalResult(message=msg, score=point.score))
        return results

    async def clear(self, session_id: str) -> None:
        if not self._ready:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="session_id", match=qmodels.MatchValue(value=session_id)
                        )
                    ]
                )
            ),
        )

    async def close(self) -> None:
        await self._client.close()
