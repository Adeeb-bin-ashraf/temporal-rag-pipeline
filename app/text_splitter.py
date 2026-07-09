"""Text chunking utilities for the RAG pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.pdf_loader import DocumentPage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk of text along with its retrieval metadata."""

    text: str
    metadata: dict[str, Any]


class TextSplitter:
    """Split extracted text into overlapping retrieval chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(
            "Initialized text splitter with chunk_size=%s and chunk_overlap=%s",
            self.chunk_size,
            self.chunk_overlap,
        )

    def split_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[DocumentChunk]:
        """Split a document body into sentence-aware, overlapping chunks.

        Text is broken on sentence boundaries and packed into chunks up to
        ``chunk_size`` characters, so chunks end on natural boundaries instead of
        mid-word. Each chunk carries a tail of the previous one (up to
        ``chunk_overlap`` characters) to preserve context across boundaries.
        """
        normalized_text = re.sub(r"\s+", " ", text).strip()
        if not normalized_text:
            return []

        # Break into sentences, hard-splitting any that alone exceed chunk_size.
        units: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", normalized_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= self.chunk_size:
                units.append(sentence)
            else:
                units.extend(self._hard_split(sentence))

        chunk_texts: list[str] = []
        current: list[str] = []
        current_length = 0
        for unit in units:
            addition = len(unit) + (1 if current_length else 0)
            if current and current_length + addition > self.chunk_size:
                chunk_texts.append(" ".join(current))
                current, current_length = self._overlap_tail(current)
                addition = len(unit) + (1 if current_length else 0)
            current.append(unit)
            current_length += addition

        if current:
            chunk_texts.append(" ".join(current))

        chunks: list[DocumentChunk] = []
        for index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_metadata = dict(metadata or {})
            chunk_metadata.setdefault("chunk_number", index)
            chunks.append(DocumentChunk(text=chunk_text, metadata=chunk_metadata))

        logger.info("Created %s chunk(s) from text input", len(chunks))
        return chunks

    def _hard_split(self, sentence: str) -> list[str]:
        """Split an over-long sentence into word-bounded pieces under chunk_size."""
        pieces: list[str] = []
        current: list[str] = []
        current_length = 0
        for word in sentence.split():
            addition = len(word) + (1 if current_length else 0)
            if current and current_length + addition > self.chunk_size:
                pieces.append(" ".join(current))
                current, current_length = [], 0
                addition = len(word)
            current.append(word)
            current_length += addition
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _overlap_tail(self, units: list[str]) -> tuple[list[str], int]:
        """Return the trailing units (and their length) fitting in chunk_overlap."""
        tail: list[str] = []
        length = 0
        for unit in reversed(units):
            addition = len(unit) + (1 if length else 0)
            if length and length + addition > self.chunk_overlap:
                break
            tail.insert(0, unit)
            length += addition
        return tail, length

    def split_documents(self, documents: Sequence[DocumentPage], document_name: str | None = None) -> list[DocumentChunk]:
        """Split a sequence of document page records into chunk records."""
        chunks: list[DocumentChunk] = []

        for document in documents:
            page_metadata = {
                "document_name": document_name or "unknown_document",
                "page_number": document.page,
            }
            page_chunks = self.split_text(document.text, metadata=page_metadata)
            if not page_chunks:
                continue

            for chunk_index, chunk in enumerate(page_chunks, start=1):
                metadata = dict(chunk.metadata)
                metadata["chunk_number"] = chunk_index
                chunks.append(DocumentChunk(text=chunk.text, metadata=metadata))

        logger.info("Created %s chunk(s) from %s document page(s)", len(chunks), len(documents))
        return chunks
