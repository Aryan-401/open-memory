"""In-process BM25 index and Reciprocal Rank Fusion (RRF) for hybrid retrieval.

BM25 is a bag-of-words relevance function that scores each document against a query using
term frequency (TF) and inverse document frequency (IDF) with length normalisation.
Unlike cosine similarity, it handles exact keyword matches well: rare terms that appear
in few documents score much higher than common filler words.

IDF and avgdl are recomputed at query time so scores stay accurate as the corpus grows.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked lists of document IDs.

    Each document at rank r in a list earns weight 1/(k + r + 1). Weights are summed
    across all lists; documents are returned highest-score-first.

    ``k=60`` is the empirically validated default from the original RRF paper; higher k
    flattens differences between ranks (less winner-takes-all), lower k amplifies them.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class BM25Store:
    """In-process BM25 index partitioned by session_id.

    No persistence across restarts (v1) — same design principle as
    :class:`~openmemory.strategies.graph.NetworkxGraphStore`.

    Parameters
    ----------
    k1:
        Term-saturation parameter. Higher values give more weight to raw TF before the
        score saturates. Typical range 1.2–2.0; default 1.5.
    b:
        Length normalisation. ``0.0`` = no normalisation, ``1.0`` = full normalisation
        to corpus average document length. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        # session_id -> ordered list of (msg_id, token_list)
        self._docs: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    def add(self, session_id: str, msg_id: str, text: str) -> None:
        """Tokenise ``text`` and append it to the session's BM25 index."""
        tokens = _tokenize(text)
        self._docs[session_id].append((msg_id, tokens))

    def search(self, session_id: str, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(msg_id, bm25_score)`` pairs sorted by descending score.

        IDF and avgdl are computed live over the current corpus so scores remain
        consistent as new messages are added between calls.
        """
        corpus = self._docs.get(session_id, [])
        if not corpus:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        N = len(corpus)
        avgdl = sum(len(tokens) for _, tokens in corpus) / N

        # Document frequency: number of distinct docs containing each term.
        df: dict[str, int] = defaultdict(int)
        for _, tokens in corpus:
            for term in set(tokens):
                df[term] += 1

        results: list[tuple[str, float]] = []
        k1, b = self._k1, self._b
        for msg_id, tokens in corpus:
            doc_len = len(tokens)
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1

            score = 0.0
            for term in query_terms:
                if term not in df:
                    continue
                idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                freq = tf[term]
                tf_norm = freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc_len / avgdl))
                score += idf * tf_norm

            if score > 0:
                results.append((msg_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def clear(self, session_id: str) -> None:
        self._docs.pop(session_id, None)
