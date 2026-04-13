from .engine import ContextEngine, ContextEngineConfig, ContextReport
from .pricing import get_pricing, PRESETS
from .wrappers import wrap_openai, WrappedClient
from .retriever import Retriever
from .pipeline import Pipeline
from .store.memory import MemoryStore, SearchResult
from .store.persistent import PersistentStore
from .loaders import load
from .router import Router, RouteRule, score_complexity
from .cache import EmbeddingCache, ResponseCache, CachedEmbedder
from .hybrid_scorer import HybridScorer
from .scoring import SemanticScorer
from .stemmer import stem, tokenize_and_stem

__version__ = "0.3.0"

__all__ = [
    # Core
    "ContextEngine",
    "ContextEngineConfig",
    "ContextReport",
    # Scoring
    "HybridScorer",
    "SemanticScorer",
    # Stemmer
    "stem",
    "tokenize_and_stem",
    # Loaders
    "load",
    # Store
    "MemoryStore",
    "PersistentStore",
    "SearchResult",
    # Retriever
    "Retriever",
    # Pipeline
    "Pipeline",
    # Router
    "Router",
    "RouteRule",
    "score_complexity",
    # Cache
    "EmbeddingCache",
    "ResponseCache",
    "CachedEmbedder",
    # Wrappers
    "wrap_openai",
    "WrappedClient",
    # Pricing
    "get_pricing",
    "PRESETS",
    # Version
    "__version__",
]
