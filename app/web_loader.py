"""Web page loading utilities for the RAG pipeline.

Mirrors ``PDFLoader``: fetches a URL, extracts the readable main text, and
returns it as ``DocumentPage`` records so the exact same split → embed → store
pipeline can index web content just like a PDF.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.pdf_loader import DocumentPage

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; TemporalRAG/1.0; +https://example.com/bot)"

# Structural / non-content tags that pollute extracted text.
_STRIP_TAGS = (
    "script", "style", "noscript", "template", "nav", "footer", "header",
    "aside", "form", "svg", "iframe", "button",
)


class WebLoader:
    """Fetch and normalize the readable text of a web page."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout
        logger.debug("Initialized web loader with timeout=%ss", self.timeout)

    def load_url(self, url: str) -> list[DocumentPage]:
        """Fetch a URL and return its extracted text as a single page."""
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme '{parsed.scheme or '(none)'}': use http(s)://")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise RuntimeError(f"Failed to fetch URL {url}: {exc}") from exc

        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            raise ValueError(f"URL did not return HTML/text (content-type: {content_type or 'unknown'})")

        title, text = self._extract(response.text)
        if not text:
            raise ValueError(f"No readable text could be extracted from {url}")

        logger.info("Fetched %s character(s) of text from %s", len(text), url)
        return [DocumentPage(page=1, text=text)]

    @staticmethod
    def _extract(html: str) -> tuple[str, str]:
        """Return (title, normalized_body_text) from raw HTML."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""
        container = soup.find("main") or soup.find("article") or soup.body or soup
        raw_text = container.get_text(separator=" ")
        normalized = re.sub(r"\s+", " ", raw_text).strip()
        return title, normalized
