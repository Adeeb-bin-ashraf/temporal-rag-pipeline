"""Temporal activities for the RAG pipeline.

Each activity wraps one business responsibility so the workflow can orchestrate
reliably in Temporal without duplicating the core RAG logic.
"""

from __future__ import annotations

import logging
from typing import Any

from temporalio import activity

from app.embeddings import EmbeddingModel
from app.pdf_loader import DocumentPage, PDFLoader
from app.rag_service import RAGService
from app.reranker import Reranker
from app.text_splitter import DocumentChunk, TextSplitter
from app.vector_store import SearchResult, VectorStore
from app.web_loader import WebLoader

logger = logging.getLogger(__name__)


def _raise_activity_error(activity_name: str, detail: str, exc: Exception) -> None:
    """Log a structured error and re-raise it as a runtime failure."""
    logger.exception("%s failed: %s", activity_name, detail)
    raise RuntimeError(f"{activity_name} failed: {detail}: {exc}") from exc


@activity.defn
async def read_pdf_activity(pdf_path: str) -> list[DocumentPage]:
    """Read and normalize PDF text so the workflow has structured page content."""
    # This activity isolates document ingestion from orchestration, making retries
    # and failures easier to reason about when the PDF is unreadable or missing.
    logger.info("Activity started: read_pdf for %s", pdf_path)
    try:
        pdf_loader = PDFLoader()
        return pdf_loader.load_document(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("read_pdf", f"Unable to read PDF {pdf_path}", exc)


@activity.defn
async def read_url_activity(url: str) -> list[DocumentPage]:
    """Fetch and normalize web page text so the workflow has structured content."""
    # This activity isolates web ingestion from orchestration, mirroring
    # read_pdf so the same split/embed/store steps can index URLs and PDFs alike.
    logger.info("Activity started: read_url for %s", url)
    try:
        web_loader = WebLoader()
        return web_loader.load_url(url)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("read_url", f"Unable to read URL {url}", exc)


@activity.defn
async def split_document_activity(pages: list[DocumentPage], document_name: str) -> list[DocumentChunk]:
    """Split document pages into overlapping chunks for retrieval."""
    # This activity keeps chunking as a discrete step so the workflow can retry
    # or inspect failures without reprocessing the whole document pipeline.
    logger.info("Activity started: split_document for %s", document_name)
    try:
        text_splitter = TextSplitter()
        return text_splitter.split_documents(pages, document_name=document_name)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("split_document", f"Unable to split document {document_name}", exc)


@activity.defn
async def generate_embeddings_activity(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Generate embeddings for document chunks before indexing them."""
    # This activity isolates the expensive embedding step so the workflow can
    # retry it independently when model inference is transiently unavailable.
    logger.info("Activity started: generate_embeddings for %s chunk(s)", len(chunks))
    try:
        embedding_model = EmbeddingModel()
        texts = [chunk.text for chunk in chunks]
        return embedding_model.embed_documents(texts)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("generate_embeddings", "Embedding generation failed", exc)


@activity.defn
async def store_vectors_activity(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> dict[str, Any]:
    """Store embeddings and chunk metadata in Qdrant for later retrieval."""
    # This activity encapsulates the vector database write so the workflow can
    # safely retry the persistence step without re-running ingestion.
    logger.info("Activity started: store_vectors for %s chunk(s)", len(chunks))
    try:
        vector_store = VectorStore()
        inserted = vector_store.add_documents(chunks, embeddings)
        return {"inserted": inserted}
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("store_vectors", "Vector storage failed", exc)


@activity.defn
async def generate_query_embedding_activity(question: str) -> list[float]:
    """Generate an embedding for the incoming question."""
    # This activity ensures the query vectorization step is retried separately
    # from retrieval and answer generation in the question workflow.
    logger.info("Activity started: generate_query_embedding")
    try:
        embedding_model = EmbeddingModel()
        return embedding_model.embed_query(question)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("generate_query_embedding", "Query embedding generation failed", exc)


@activity.defn
async def retrieve_chunks_activity(query_vector: list[float], limit: int = 4) -> list[SearchResult]:
    """Retrieve relevant chunks from the vector store for grounding."""
    # This activity isolates vector retrieval so the workflow can retry search
    # independently when the database is briefly unavailable.
    logger.info("Activity started: retrieve_chunks with limit=%s", limit)
    try:
        vector_store = VectorStore()
        return vector_store.search(query_vector, limit=limit)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("retrieve_chunks", "Chunk retrieval failed", exc)


@activity.defn
async def rerank_chunks_activity(
    question: str, search_results: list[SearchResult], top_n: int = 4
) -> list[SearchResult]:
    """Rerank retrieved candidates with a cross-encoder and keep the best few."""
    # This activity isolates reranking so it can be retried independently and so
    # the sharper relevance ordering is applied before answer generation.
    logger.info("Activity started: rerank_chunks (%s candidate(s) -> top %s)", len(search_results), top_n)
    try:
        reranker = Reranker()
        return reranker.rerank(question, search_results, top_n=top_n)
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("rerank_chunks", "Chunk reranking failed", exc)


@activity.defn
async def generate_answer_activity(question: str, search_results: list[SearchResult]) -> dict[str, Any]:
    """Generate a grounded answer using the existing RAG service and Ollama."""
    # This activity keeps LLM generation as a single step so the workflow can
    # preserve the retrieval context and retry generation safely.
    logger.info("Activity started: generate_answer for question=%s", question)
    try:
        rag_service = RAGService()
        context = rag_service._build_context(search_results)
        prompt = rag_service._build_prompt(question, context)
        response = rag_service.ollama_client.generate(
            model=rag_service.settings.ollama_model,
            prompt=prompt,
            stream=False,
        )
        answer = rag_service._extract_answer_text(response)
        return {"answer": answer, "context_chunks": [item.text for item in search_results]}
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        _raise_activity_error("generate_answer", "Answer generation failed", exc)
