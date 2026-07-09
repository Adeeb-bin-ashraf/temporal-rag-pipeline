"""Vector storage and retrieval utilities for the RAG pipeline."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings
from app.text_splitter import DocumentChunk

logger = logging.getLogger(__name__)

# Stable namespace for deriving deterministic point IDs (see _deterministic_id).
_POINT_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def _deterministic_id(chunk: "DocumentChunk") -> str:
    """Derive a stable point ID from a chunk's source and text.

    Re-indexing the same document therefore upserts the same points instead of
    accumulating duplicate vectors on every run.
    """
    source = str(chunk.metadata.get("document_name", ""))
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{source}::{chunk.text}"))


@dataclass(frozen=True)
class SearchResult:
    """Represents a retrieved vector match."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore:
    """Store and retrieve vectors in Qdrant."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        settings = get_settings()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.client = client or QdrantClient(host=self.host, port=self.port, timeout=30)
        logger.debug(
            "Initialized vector store for %s:%s collection=%s",
            self.host,
            self.port,
            self.collection_name,
        )

    def create_collection(self, vector_size: int | None = None) -> None:
        """Create the configured Qdrant collection if it does not already exist."""
        if vector_size is None:
            raise ValueError("vector_size must be provided to create a collection")

        try:
            collections = self.client.get_collections().collections
            if any(collection.name == self.collection_name for collection in collections):
                logger.info("Collection already exists: %s", self.collection_name)
                return

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logger.info("Created collection %s with vector size %s", self.collection_name, vector_size)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Failed to create Qdrant collection: {exc}") from exc

    def add_documents(self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]) -> int:
        """Insert chunks and embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        if not chunks:
            return 0

        vector_size = len(embeddings[0])
        self.create_collection(vector_size=vector_size)

        points: list[models.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            points.append(
                models.PointStruct(
                    id=_deterministic_id(chunk),
                    vector=list(embedding),
                    payload={"text": chunk.text, "metadata": chunk.metadata},
                )
            )

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("Inserted %s vector(s) into Qdrant", len(points))
            return len(points)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Failed to insert vectors into Qdrant: {exc}") from exc

    def search(self, query_vector: list[float], limit: int = 5) -> list[SearchResult]:
        """Search the vector store for the nearest neighbors."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            hits = getattr(response, "points", [])
            results = [
                SearchResult(
                    id=str(hit.id),
                    score=float(hit.score),
                    text=str(hit.payload.get("text", "")),
                    metadata=dict(hit.payload.get("metadata", {})),
                )
                for hit in hits
            ]
            logger.info("Retrieved %s search result(s)", len(results))
            return results
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc

    def delete_collection(self) -> None:
        """Delete the configured collection from Qdrant."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info("Deleted collection %s", self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Failed to delete Qdrant collection: {exc}") from exc

    def count(self) -> int:
        """Return the number of stored vectors."""
        try:
            result = self.client.count(collection_name=self.collection_name)
            return int(result.count)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Failed to count Qdrant vectors: {exc}") from exc
