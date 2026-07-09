"""Embedding utilities for the RAG pipeline."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Generate embeddings with a sentence-transformer model."""

    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = self._load_model(self.model_name)
        logger.info("Embedding model ready: %s", self.model_name)

    @classmethod
    def _load_model(cls, model_name: str) -> SentenceTransformer:
        """Load the embedding model once per model name."""
        if model_name not in cls._model_cache:
            logger.info("Loading embedding model: %s", model_name)
            cls._model_cache[model_name] = SentenceTransformer(model_name)
        return cls._model_cache[model_name]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Create embeddings for a collection of document texts."""
        if not documents:
            return []
        try:
            embeddings = self._model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
            logger.info("Generated embeddings for %s document(s)", len(documents))
            return embeddings.tolist()
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        """Create an embedding for a single query string."""
        if not query.strip():
            raise ValueError("Query text cannot be empty")
        try:
            embedding = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            logger.info("Generated embedding for query")
            return embedding.tolist()
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Query embedding generation failed: {exc}") from exc
