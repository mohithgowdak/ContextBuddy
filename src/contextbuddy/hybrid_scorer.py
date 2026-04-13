"""
HybridScorer -- a zero-dependency relevance scorer that combines four signals:

1. **BM25**          – the algorithm behind Elasticsearch / Lucene.  Handles
                       term-frequency saturation, doc-length normalization,
                       and inverse-document-frequency weighting.
2. **Stemming**      – "payments" matches "payment", "running" matches "run".
3. **Synonym expansion** – "car" matches "automobile", "purchase" matches "buy".
4. **Character n-gram overlap** – catches typos and morphological variants that
                       stemming misses ("optimise" vs "optimize").

All in pure Python, zero dependencies, runs offline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .stemmer import stem, tokenize_and_stem
from .synonyms import expand_synonyms, expand_query_terms

_word_re = re.compile(r"[a-z0-9]+")

# ── Stop words (small, high-frequency words that add noise to BM25) ──
_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "but", "or", "nor", "not", "so", "if", "then", "than",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over",
    "no", "yes", "all", "each", "every", "both", "few", "more",
    "other", "some", "such", "only", "own", "same", "too", "very",
    "just", "also", "how", "when", "where", "why",
    "here", "there", "out", "as",
})


def _tokenize_raw(text: str) -> List[str]:
    """Lowercase word tokens, no stemming."""
    return [m.group(0) for m in _word_re.finditer(text.lower())]


def _remove_stops(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ── Character n-gram helpers ──

def _char_ngrams(word: str, n: int = 3) -> Set[str]:
    """Return the set of character n-grams for a word."""
    if len(word) < n:
        return {word}
    return {word[i : i + n] for i in range(len(word) - n + 1)}


def _ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity over character n-grams of two words."""
    ga = _char_ngrams(a, n)
    gb = _char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


# ── BM25 core ──

def _compute_idf(term: str, doc_freqs: Dict[str, int], n_docs: int) -> float:
    """Okapi BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)."""
    df = doc_freqs.get(term, 0)
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_term_score(
    tf: float,
    idf: float,
    doc_len: int,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    num = tf * (k1 + 1.0)
    denom = tf + k1 * (1.0 - b + b * (doc_len / avg_dl)) if avg_dl > 0 else tf + k1
    return idf * (num / denom)


@dataclass
class HybridScorer:
    """
    Drop-in replacement for SemanticScorer with dramatically better
    relevance scoring.  Same interface: ``score(query=..., chunks=...) -> List[float]``.

    Combines:
    - BM25 (keyword relevance with IDF weighting)
    - Stemming (word-form normalization)
    - Synonym expansion (built-in thesaurus)
    - Character n-gram fuzzy matching

    Zero dependencies, pure Python, runs offline.
    """

    bm25_weight: float = 0.70
    ngram_weight: float = 0.15
    synonym_weight: float = 0.15
    k1: float = 1.5
    b: float = 0.75
    ngram_n: int = 3
    ngram_threshold: float = 0.35

    def score(self, *, query: str, chunks: Sequence[str]) -> List[float]:
        if not chunks:
            return []

        query_raw = _remove_stops(_tokenize_raw(query))
        query_stemmed = _remove_stops(tokenize_and_stem(query))

        # Synonym expansion on raw (unstemmed) query terms
        query_expanded: Set[str] = set(query_raw)
        synonym_additions: Set[str] = set()
        for t in query_raw:
            syns = expand_synonyms(t)
            synonym_additions.update(syns - {t})
        query_expanded.update(synonym_additions)

        # Also stem the synonym expansions
        query_stemmed_expanded: Set[str] = set(query_stemmed)
        for t in query_expanded:
            query_stemmed_expanded.add(stem(t))

        # Tokenize + stem all chunks
        chunk_tokens_raw: List[List[str]] = []
        chunk_tokens_stemmed: List[List[str]] = []
        chunk_tfs: List[Counter] = []

        for ch in chunks:
            raw = _remove_stops(_tokenize_raw(ch))
            stemmed = _remove_stops(tokenize_and_stem(ch))
            chunk_tokens_raw.append(raw)
            chunk_tokens_stemmed.append(stemmed)
            chunk_tfs.append(Counter(stemmed))

        n_docs = len(chunks)
        avg_dl = sum(len(t) for t in chunk_tokens_stemmed) / n_docs if n_docs else 1.0

        # Document frequency for each stemmed term
        doc_freqs: Counter = Counter()
        for tf in chunk_tfs:
            for term in tf:
                doc_freqs[term] += 1

        # Also count doc-freq for synonym-expanded terms
        for term in query_stemmed_expanded:
            if term not in doc_freqs:
                for tf in chunk_tfs:
                    if term in tf:
                        doc_freqs[term] += 1

        # ── Score each chunk ──
        bm25_scores: List[float] = []
        ngram_scores: List[float] = []
        synonym_scores: List[float] = []

        for i, ch in enumerate(chunks):
            tf = chunk_tfs[i]
            doc_len = len(chunk_tokens_stemmed[i])

            # --- BM25 on stemmed terms ---
            bm25 = 0.0
            for qt in query_stemmed:
                idf = _compute_idf(qt, doc_freqs, n_docs)
                bm25 += _bm25_term_score(
                    tf=tf.get(qt, 0),
                    idf=idf,
                    doc_len=doc_len,
                    avg_dl=avg_dl,
                    k1=self.k1,
                    b=self.b,
                )
            bm25_scores.append(bm25)

            # --- Synonym match bonus ---
            syn_hits = 0
            raw_set = set(chunk_tokens_raw[i])
            stemmed_set = set(chunk_tokens_stemmed[i])
            for syn_term in synonym_additions:
                stemmed_syn = stem(syn_term)
                if syn_term in raw_set or stemmed_syn in stemmed_set:
                    syn_hits += 1
            syn_score = syn_hits / max(len(synonym_additions), 1)
            synonym_scores.append(syn_score)

            # --- Character n-gram fuzzy matching ---
            ngram_total = 0.0
            ngram_count = 0
            for qt in query_raw:
                best_sim = 0.0
                for dt in chunk_tokens_raw[i]:
                    sim = _ngram_similarity(qt, dt, n=self.ngram_n)
                    if sim > best_sim:
                        best_sim = sim
                    if best_sim >= 0.9:
                        break
                if best_sim >= self.ngram_threshold:
                    ngram_total += best_sim
                    ngram_count += 1
            ngram_score = ngram_total / max(len(query_raw), 1)
            ngram_scores.append(ngram_score)

        # ── Normalize each signal to [0, 1] ──
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        if bm25_max <= 0:
            bm25_max = 1.0
        bm25_norm = [s / bm25_max for s in bm25_scores]

        ngram_max = max(ngram_scores) if ngram_scores else 1.0
        if ngram_max <= 0:
            ngram_max = 1.0
        ngram_norm = [s / ngram_max for s in ngram_scores]

        syn_max = max(synonym_scores) if synonym_scores else 1.0
        if syn_max <= 0:
            syn_max = 1.0
        syn_norm = [s / syn_max for s in synonym_scores]

        # ── Weighted combination ──
        combined = [
            self.bm25_weight * bm25_norm[i]
            + self.ngram_weight * ngram_norm[i]
            + self.synonym_weight * syn_norm[i]
            for i in range(n_docs)
        ]

        return combined
