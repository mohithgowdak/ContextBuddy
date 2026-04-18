# Scoring (how chunks are ranked)

ContextBuddy separates **scoring** (“how relevant is this chunk?”) from **embeddings** (“how do we represent text?”).

## Default: `HybridScorer` (zero-dep)

Defined in `src/contextbuddy/hybrid_scorer.py`.

It combines:

- **BM25** (IR baseline used by search engines)
- **Stemming**
- **Synonym expansion**
- **Character n-gram fuzzy matching**
- **Intent-aware boosts** (numbers/method/related-work) using universal signals:
  - numeric density for “results/metrics”
  - “PROPOSED METHODOLOGY / step” cues for “how it works”
  - citation density + “RELATED WORKS” cues for “previous work”

This is still deterministic and dependency-free.

## Embedding-based: `SemanticScorer`

Defined in `src/contextbuddy/scoring.py`. It computes cosine similarity between:

- query embedding
- chunk embeddings

