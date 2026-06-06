"""Tests for BM25Store, SparseHybridMemory, and FullHybridMemory.

Focus: mechanics (RRF ordering, BM25 scoring, dedup, session isolation, chronological
output), not semantic-vs-lexical contrast — FakeEmbedder is also bag-of-words, so it
cannot distinguish the two retrieval channels; that distinction is covered by BM25Store's
own unit tests and the integration shape tests below.
"""

from __future__ import annotations

from openmemory.storage.bm25_store import BM25Store, rrf
from openmemory.strategies.full_hybrid import FullHybridMemory  # noqa: E402
from openmemory.strategies.sparse_hybrid import SparseHybridMemory  # noqa: E402

# ---------------------------------------------------------------------------
# rrf() unit tests (pure function, no async)
# ---------------------------------------------------------------------------

def test_rrf_consensus_ranks_highest():
    """A doc appearing first in both lists should outrank docs in only one."""
    r1 = ["X", "Y", "Z"]
    r2 = ["X", "W", "Z"]
    result = rrf([r1, r2], k=60)
    # X is #1 in both: 2 * (1/61) = 0.0328
    # Z is #3 in both: 2 * (1/63) = 0.0317
    # Y is #2 in r1 only: 1/62 = 0.0161
    # W is #2 in r2 only: 1/62 = 0.0161
    assert result[0] == "X"
    assert result.index("Z") < result.index("Y")  # dual-list presence beats single-list


def test_rrf_single_ranking_is_passthrough():
    """Single input ranking → same order (plus dedup, which is irrelevant here)."""
    ranking = ["A", "B", "C", "D"]
    result = rrf([ranking], k=60)
    assert result == ranking


def test_rrf_deduplicates_across_lists():
    """The same doc ID in multiple lists should appear only once in output."""
    r1 = ["A", "B"]
    r2 = ["B", "A"]
    result = rrf([r1, r2], k=60)
    assert len(result) == len(set(result))


def test_rrf_empty_lists():
    result = rrf([[], []], k=60)
    assert result == []

    result = rrf([["A"], []], k=60)
    assert result == ["A"]


# ---------------------------------------------------------------------------
# BM25Store unit tests
# ---------------------------------------------------------------------------

def test_bm25_exact_keyword_scores_above_zero():
    store = BM25Store()
    store.add("s", "a", "quantum physics and field theory")
    store.add("s", "b", "cooking recipes and ingredients")

    hits = store.search("s", "quantum", k=5)
    ids = [h[0] for h in hits]
    assert "a" in ids
    assert "b" not in ids  # no query term match → score 0, excluded


def test_bm25_idf_updates_as_corpus_grows():
    """Rare terms score higher than common terms — and IDF rises when docs are added."""
    store = BM25Store()
    # Add three docs all containing "common"
    store.add("s", "a", "common word appears here")
    store.add("s", "b", "common word again")
    store.add("s", "c", "common word one more time")
    hits_before = store.search("s", "common", k=3)
    score_before = max(s for _, s in hits_before)

    # Add two docs without "common" → IDF for "common" increases (it's now in 3/5 docs)
    store.add("s", "d", "something entirely different topic")
    store.add("s", "e", "another unrelated subject matter")
    hits_after = store.search("s", "common", k=5)
    score_after = max(s for _, s in hits_after)

    assert score_after > score_before


def test_bm25_rare_term_wins_over_common_term():
    """Query for a term that is rare in the corpus should prefer the doc with that term."""
    store = BM25Store()
    store.add("s", "rare_doc", "palimpsest ancient manuscripts overlay")
    store.add("s", "common_a", "the cat sat on the mat")
    store.add("s", "common_b", "the dog ran in the park")
    store.add("s", "common_c", "the bird flew over the house")

    hits = store.search("s", "palimpsest", k=4)
    # Only rare_doc contains "palimpsest"; all others have zero overlap
    assert hits[0][0] == "rare_doc"


def test_bm25_empty_corpus_returns_empty():
    store = BM25Store()
    assert store.search("no_session", "anything", k=5) == []


def test_bm25_empty_query_returns_empty():
    store = BM25Store()
    store.add("s", "a", "hello world")
    assert store.search("s", "", k=5) == []


def test_bm25_clear_removes_session():
    store = BM25Store()
    store.add("s", "a", "hello world")
    store.clear("s")
    assert store.search("s", "hello", k=5) == []


def test_bm25_sessions_isolated():
    store = BM25Store()
    store.add("alpha", "a1", "skiing mountains alps")
    store.add("beta", "b1", "cooking pasta carbonara")

    alpha_hits = store.search("alpha", "skiing", k=5)
    assert all(msg_id.startswith("a") for msg_id, _ in alpha_hits)

    beta_hits = store.search("beta", "skiing", k=5)
    assert beta_hits == []  # "skiing" not in beta's corpus


# ---------------------------------------------------------------------------
# SparseHybridMemory integration tests
# ---------------------------------------------------------------------------

async def test_sparse_hybrid_aadd_and_context(store, embedder, vectors):
    bm25 = BM25Store()
    mem = SparseHybridMemory("s1", store, embedder, vectors, bm25, top_k=3)

    await mem.aadd({"role": "user", "content": "I enjoy hiking in the Alps"})
    await mem.aadd({"role": "user", "content": "my cat is named Whiskers"})

    ctx = await mem.aget_context(query="Alps hiking")
    assert len(ctx) >= 1
    assert any("Alps" in m.content or "hiking" in m.content for m in ctx)


async def test_sparse_hybrid_no_query_returns_recent(store, embedder, vectors):
    bm25 = BM25Store()
    mem = SparseHybridMemory("s1", store, embedder, vectors, bm25, top_k=2)

    for i in range(5):
        await mem.aadd({"role": "user", "content": f"message {i}"})

    ctx = await mem.aget_context()  # no query
    # Falls back to recency (last top_k=2 messages)
    assert len(ctx) == 2
    assert ctx[-1].content == "message 4"


async def test_sparse_hybrid_output_chronological(store, embedder, vectors):
    bm25 = BM25Store()
    mem = SparseHybridMemory("s1", store, embedder, vectors, bm25, top_k=5)

    await mem.aadd({"role": "user", "content": "first message hiking trails"})
    await mem.aadd({"role": "user", "content": "second message hiking mountains"})
    await mem.aadd({"role": "user", "content": "third message hiking gear"})

    ctx = await mem.aget_context(query="hiking")
    timestamps = [m.timestamp for m in ctx]
    assert timestamps == sorted(timestamps)


async def test_sparse_hybrid_no_duplicate_ids(store, embedder, vectors):
    bm25 = BM25Store()
    mem = SparseHybridMemory("s1", store, embedder, vectors, bm25, top_k=5)

    for i in range(6):
        await mem.aadd({"role": "user", "content": f"hiking topic {i}"})

    ctx = await mem.aget_context(query="hiking")
    ids = [m.id for m in ctx]
    assert len(ids) == len(set(ids))


async def test_sparse_hybrid_sessions_isolated(store, embedder, vectors):
    bm25 = BM25Store()
    alpha = SparseHybridMemory("alpha", store, embedder, vectors, bm25, top_k=5)
    beta = SparseHybridMemory("beta", store, embedder, vectors, bm25, top_k=5)

    await alpha.aadd({"role": "user", "content": "alpha skiing mountains"})
    await beta.aadd({"role": "user", "content": "beta cooking recipes"})

    ctx = await alpha.aget_context(query="skiing")
    assert all(m.session_id == "alpha" for m in ctx)

    ctx2 = await beta.aget_context(query="skiing")
    assert all(m.session_id == "beta" for m in ctx2)


async def test_sparse_hybrid_clear(store, embedder, vectors):
    bm25 = BM25Store()
    mem = SparseHybridMemory("s1", store, embedder, vectors, bm25, top_k=5)

    await mem.aadd({"role": "user", "content": "hiking in mountains"})
    await mem.aclear()

    ctx = await mem.aget_context(query="hiking")
    assert ctx == []


# ---------------------------------------------------------------------------
# FullHybridMemory integration tests
# ---------------------------------------------------------------------------

async def test_full_hybrid_surfaces_old_and_recent(store, embedder, vectors):
    """Old relevant msg surfaces via BM25+vector; recent unrelated msg via window."""
    bm25 = BM25Store()
    mem = FullHybridMemory(
        "s1", store, embedder, vectors, bm25, window_size=2, top_k=4
    )

    await mem.aadd({"role": "user", "content": "I am planning a trip to the Alps"})
    for i in range(4):
        await mem.aadd({"role": "user", "content": f"unrelated topic {i}"})

    ctx = await mem.aget_context(query="Alps trip planning")
    contents = [m.content for m in ctx]

    assert any("Alps" in c for c in contents)          # retrieved by BM25/vector
    assert any("unrelated topic 3" in c for c in contents)  # in window


async def test_full_hybrid_no_duplicate_ids(store, embedder, vectors):
    bm25 = BM25Store()
    mem = FullHybridMemory("s1", store, embedder, vectors, bm25, window_size=3, top_k=5)

    for i in range(6):
        await mem.aadd({"role": "user", "content": f"hiking topic {i}"})

    ctx = await mem.aget_context(query="hiking")
    ids = [m.id for m in ctx]
    assert len(ids) == len(set(ids))


async def test_full_hybrid_no_query_returns_window(store, embedder, vectors):
    bm25 = BM25Store()
    mem = FullHybridMemory("s1", store, embedder, vectors, bm25, window_size=2, top_k=5)

    for i in range(5):
        await mem.aadd({"role": "user", "content": f"turn {i}"})

    ctx = await mem.aget_context()  # no query
    assert len(ctx) == 2
    assert ctx[-1].content == "turn 4"


async def test_full_hybrid_chronological_output(store, embedder, vectors):
    bm25 = BM25Store()
    mem = FullHybridMemory("s1", store, embedder, vectors, bm25, window_size=3, top_k=5)

    await mem.aadd({"role": "user", "content": "first hiking message"})
    await mem.aadd({"role": "user", "content": "second hiking message"})
    await mem.aadd({"role": "user", "content": "third hiking message"})

    ctx = await mem.aget_context(query="hiking")
    timestamps = [m.timestamp for m in ctx]
    assert timestamps == sorted(timestamps)


async def test_full_hybrid_sessions_isolated(store, embedder, vectors):
    bm25 = BM25Store()
    alpha = FullHybridMemory("alpha", store, embedder, vectors, bm25)
    beta = FullHybridMemory("beta", store, embedder, vectors, bm25)

    await alpha.aadd({"role": "user", "content": "alpha hiking mountains"})
    await beta.aadd({"role": "user", "content": "beta cooking pasta"})

    ctx = await alpha.aget_context(query="hiking")
    assert all(m.session_id == "alpha" for m in ctx)


async def test_full_hybrid_clear(store, embedder, vectors):
    bm25 = BM25Store()
    mem = FullHybridMemory("s1", store, embedder, vectors, bm25, top_k=5)

    await mem.aadd({"role": "user", "content": "hiking in mountains"})
    await mem.aclear()

    ctx = await mem.aget_context(query="hiking")
    assert ctx == []
