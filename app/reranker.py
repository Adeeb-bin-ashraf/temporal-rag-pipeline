"""Cross-encoder reranking for the RAG pipeline.

Dense vector search is fast but approximate: it ranks by embedding cosine
similarity, which can surface loosely-related chunks. A cross-encoder re-scores
each (question, chunk) pair jointly, giving a much sharper relevance ordering.
The pipeline retrieves a wide candidate set, then reranks down to the few chunks
actually sent to the LLM.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import CrossEncoder

from app.constants import DEFAULT_RERANKER_MODEL
from app.vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks with a cross-encoder model."""

    _model_cache: dict[str, CrossEncoder] = {}

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or DEFAULT_RERANKER_MODEL
        self._model = self._load_model(self.model_name)
        logger.info("Reranker ready: %s", self.model_name)

    @classmethod
    def _load_model(cls, model_name: str) -> CrossEncoder:
        """Load the cross-encoder once per model name."""
        if model_name not in cls._model_cache:
            logger.info("Loading reranker model: %s", model_name)
            cls._model_cache[model_name] = CrossEncoder(model_name)
        return cls._model_cache[model_name]

    def rerank(self, query: str, results: Sequence[SearchResult], top_n: int = 4) -> list[SearchResult]:
        """Return the ``top_n`` results reordered by cross-encoder relevance.

        The original vector-similarity score is preserved on each result for
        display; only the ordering and selection change.
        """
        if not results:
            return []
        if len(results) <= 1:
            return list(results)[:top_n]

        pairs = [(query, result.text) for result in results]
        scores = self._model.predict(pairs)
        ordered = sorted(zip(results, scores), key=lambda pair: float(pair[1]), reverse=True)
        top = [result for result, _ in ordered[:top_n]]
        logger.info("Reranked %s candidate(s) down to %s", len(results), len(top))
        return top
