"""Lightweight integration script for the RAG pipeline.

This script exercises the core RAG modules end-to-end using a sample PDF from the
configured documents directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.rag_service import RAGService
from app.sample_pdf import create_sample_pdf

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for the integration script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create the configured Qdrant collection if it does not exist."""
    try:
        collections = client.get_collections().collections
    except Exception as exc:  # pragma: no cover - defensive error wrapper
        raise RuntimeError(f"Unable to connect to Qdrant: {exc}") from exc

    if any(collection.name == collection_name for collection in collections):
        logger.info("Qdrant collection already exists: %s", collection_name)
        return

    logger.info("Creating Qdrant collection: %s", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def find_sample_pdf(documents_directory: Path) -> Path:
    """Return the first PDF file in the documents directory."""
    pdf_files = sorted(documents_directory.glob("*.pdf"))
    if not pdf_files:
        logger.info("Creating a sample PDF in %s", documents_directory)
        return create_sample_pdf(documents_directory / "sample.pdf")
    return pdf_files[0]


def main() -> None:
    """Run the integration pipeline against a sample PDF."""
    configure_logging()
    settings = get_settings()

    documents_directory = Path(settings.documents_directory)
    if not documents_directory.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_directory}")

    logger.info("Using documents directory: %s", documents_directory)

    try:
        sample_pdf = find_sample_pdf(documents_directory)
    except FileNotFoundError as exc:
        logger.error("No sample PDF available: %s", exc)
        return

    logger.info("Using sample PDF: %s", sample_pdf)

    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
        ensure_collection(client, settings.qdrant_collection, vector_size=384)
    except Exception as exc:
        logger.exception("Qdrant setup failed: %s", exc)
        return

    try:
        rag_service = RAGService()
        logger.info("Indexing document...")
        index_result = rag_service.index_pdf(sample_pdf)
        logger.info("Indexing complete: %s", index_result)

        print(f"Number of pages: {index_result['pages']}")
        print(f"Number of chunks created: {index_result['chunks']}")
        print(f"Number of vectors inserted: {index_result['inserted']}")

        question = "What is this document about?"
        logger.info("Asking question: %s", question)
        answer_result = rag_service.answer_question(question)

        print("Retrieved chunks:")
        for chunk in answer_result.get("context_chunks", []):
            print(f"- {chunk}")

        print("Final Ollama response:")
        print(answer_result.get("answer", ""))

    except Exception as exc:
        logger.exception("RAG pipeline integration failed: %s", exc)


if __name__ == "__main__":
    main()
