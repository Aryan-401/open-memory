"""Durable :class:`SessionStore` backed by SQLite via ``aiosqlite``.

Messages are stored as JSON blobs (preserving every internal field) with an
auto-incrementing ``seq`` to guarantee insertion ordering. A composite index on
``(session_id, seq)`` keeps per-session reads fast.
"""

from __future__ import annotations

import asyncio

import aiosqlite

from ..core.models import Message
from .session_store import SessionStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, seq);
"""


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session store. Durable across process restarts."""

    def __init__(self, path: str = "openmemory.db") -> None:
        self._path = path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            async with self._lock:
                if self._db is None:
                    self._db = await aiosqlite.connect(self._path)
                    await self._db.executescript(_SCHEMA)
                    await self._db.commit()
        return self._db

    async def append(self, session_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        db = await self._conn()
        await db.executemany(
            "INSERT INTO messages (session_id, message_id, payload) VALUES (?, ?, ?)",
            [(session_id, m.id, m.model_dump_json()) for m in messages],
        )
        await db.commit()

    async def get_all(self, session_id: str) -> list[Message]:
        db = await self._conn()
        async with db.execute(
            "SELECT payload FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [Message.model_validate_json(r[0]) for r in rows]

    async def recent(self, session_id: str, n: int) -> list[Message]:
        db = await self._conn()
        async with db.execute(
            "SELECT payload FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
            (session_id, n),
        ) as cur:
            rows = await cur.fetchall()
        return [Message.model_validate_json(r[0]) for r in reversed(list(rows))]

    async def clear(self, session_id: str) -> None:
        db = await self._conn()
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.commit()

    async def sessions(self) -> list[str]:
        db = await self._conn()
        async with db.execute("SELECT DISTINCT session_id FROM messages") as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
