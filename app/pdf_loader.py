"""PDF document loading utilities for the RAG pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentPage:
    """Represents a page of extracted PDF text."""

    page: int
    text: str


class PDFLoader:
    """Load and normalize PDF documents from disk."""

    def __init__(self, documents_directory: str | Path | None = None) -> None:
        self.documents_directory: Path = Path(documents_directory) if documents_directory is not None else Path("documents")
        logger.debug("Initialized PDF loader for %s", self.documents_directory)

    def load_document(self, path: str | Path) -> list[DocumentPage]:
        """Load a single PDF file and return its extracted pages."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Expected a file path for PDF loading: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            reader = PdfReader(str(file_path))
            if not reader.pages:
                raise ValueError(f"PDF contains no pages: {file_path}")

            pages: list[DocumentPage] = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = self._normalize_text(page.extract_text() or "")
                if page_text:
                    pages.append(DocumentPage(page=page_number, text=page_text))

            if not pages:
                raise ValueError(f"PDF contains no extractable text: {file_path}")

            logger.info("Loaded PDF %s with %s page(s)", file_path.name, len(pages))
            return pages
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Failed to read PDF {file_path}: {exc}") from exc

    def load_documents(self, paths: list[str | Path] | None = None) -> list[DocumentPage]:
        """Load multiple PDF files and return their pages in order."""
        if not paths:
            raise ValueError("At least one PDF path is required")

        pages: list[DocumentPage] = []
        for path in paths:
            pages.extend(self.load_document(path))
        return pages

    def load_from_directory(self, directory: str | Path | None = None) -> list[DocumentPage]:
        """Load supported PDF files from a directory."""
        target_directory = Path(directory) if directory is not None else self.documents_directory
        if not target_directory.exists():
            raise FileNotFoundError(f"Documents directory not found: {target_directory}")
        if not target_directory.is_dir():
            raise ValueError(f"Expected a directory: {target_directory}")

        pdf_paths = sorted(target_directory.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in directory: {target_directory}")

        return self.load_documents(pdf_paths)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize line breaks and whitespace in extracted text."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()
