"""High-level orchestration for the RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ollama import Client

from app.config import get_settings
from app.constants import RERANK_CANDIDATE_LIMIT
from app.embeddings import EmbeddingModel
from app.pdf_loader import DocumentPage, PDFLoader
from app.reranker import Reranker
from app.text_splitter import DocumentChunk, TextSplitter
from app.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """Coordinate PDF ingestion, retrieval, and grounded question answering."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        pdf_loader: PDFLoader | None = None,
        text_splitter: TextSplitter | None = None,
        ollama_client: Client | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = get_settings()
        self.embedding_model = embedding_model or EmbeddingModel(model_name=self.settings.embedding_model)
        self.vector_store = vector_store or VectorStore()
        self.pdf_loader = pdf_loader or PDFLoader(documents_directory=self.settings.documents_directory)
        self.text_splitter = text_splitter or TextSplitter()
        self.ollama_client = ollama_client or Client()
        self.reranker = reranker or Reranker()
        logger.debug("Initialized RAG service")

    def index_pdf(self, path: str | Path) -> dict[str, Any]:
        """Read a PDF, split it into chunks, generate embeddings, and store them."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            logger.info("Loading PDF for indexing: %s", file_path)
            pages: list[DocumentPage] = self.pdf_loader.load_document(file_path)
            logger.info("Extracted %s page(s) from %s", len(pages), file_path.name)

            chunks: list[DocumentChunk] = self.text_splitter.split_documents(pages, document_name=file_path.name)
            if not chunks:
                raise ValueError(f"No text chunks were generated from {file_path}")
            logger.info("Created %s chunk(s) from %s", len(chunks), file_path.name)

            texts = [chunk.text for chunk in chunks]
            logger.info("Generating embeddings for %s chunk(s)", len(texts))
            embeddings = self.embedding_model.embed_documents(texts)
            logger.info("Inserting %s vector(s) into the index", len(embeddings))
            inserted_count = self.vector_store.add_documents(chunks, embeddings)

            return {
                "file": str(file_path),
                "pages": len(pages),
                "chunks": len(chunks),
                "inserted": inserted_count,
            }
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"PDF indexing failed: {exc}") from exc

    def answer_question(self, question: str, top_k: int = 4) -> dict[str, Any]:
        """Answer a user question using retrieved context and an Ollama model."""
        if not question or not question.strip():
            raise ValueError("Question text cannot be empty")

        try:
            logger.info("Generating query embedding for question")
            query_embedding = self.embedding_model.embed_query(question)
            logger.info("Retrieving up to %s candidate chunks", RERANK_CANDIDATE_LIMIT)
            candidates: list[SearchResult] = self.vector_store.search(query_embedding, limit=RERANK_CANDIDATE_LIMIT)
            logger.info("Reranking to top-%s relevant chunks", top_k)
            search_results: list[SearchResult] = self.reranker.rerank(question, candidates, top_n=top_k)

            context = self._build_context(search_results)
            prompt = self._build_prompt(question, context)
            logger.info("Prompt generated for LLM")

            response = self.ollama_client.generate(model=self.settings.ollama_model, prompt=prompt, stream=False)
            answer_text = self._extract_answer_text(response)
            logger.info("LLM response generated")

            return {
                "question": question,
                "answer": answer_text,
                "context_chunks": [result.text for result in search_results],
            }
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Question answering failed: {exc}") from exc

    def _build_context(self, search_results: list[SearchResult]) -> str:
        """Build a readable context string from retrieved search results."""
        if not search_results:
            return "No relevant context was found."

        context_parts: list[str] = []
        for index, result in enumerate(search_results, start=1):
            metadata = result.metadata or {}
            source = metadata.get("document_name", "unknown_document")
            page = metadata.get("page_number", "unknown")
            context_parts.append(f"[{index}] Source: {source}, page: {page}\n{result.text}")
        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Construct a grounded prompt for the language model."""
        return (
            "You are a precise assistant. Answer the question using ONLY the supplied context.\n"
            "Format the answer as clean, concise Markdown: short paragraphs, and use bullet "
            "points or **bold** only where they genuinely aid clarity. Do not invent facts.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "If the answer is not in the context, say plainly that the indexed sources do not "
            "contain enough information."
        )

    @staticmethod
    def _extract_answer_text(response: Any) -> str:
        """Extract a string answer from an Ollama response payload."""
        if isinstance(response, dict):
            if "response" in response:
                return str(response["response"])
            return str(response)
        if hasattr(response, "response"):
            return str(response.response)
        return str(response)
