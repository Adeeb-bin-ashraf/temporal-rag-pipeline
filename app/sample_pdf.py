"""Helpers for creating a deterministic sample PDF for local testing."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SAMPLE_TEXT = (
    "This document is about the Temporal RAG pipeline and the architecture for "
    "grounding answers with retrieved context."
)


def build_sample_pdf_bytes(text: str | None = None) -> bytes:
    """Build a minimal PDF document as bytes for local integration testing."""
    content_text = text or DEFAULT_SAMPLE_TEXT
    escaped_text = content_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 50 700 Td ({escaped_text}) Tj ET"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length 44 >>\nstream\n" + content.encode("latin-1") + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    stream = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(stream))
        stream.extend(f"{index} 0 obj\n".encode("latin-1"))
        stream.extend(obj + b"\nendobj\n")

    startxref = len(stream)
    stream.extend(b"xref\n0 6\n")
    stream.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        stream.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    stream.extend(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n")
    stream.extend(str(startxref).encode("latin-1"))
    stream.extend(b"\n%%EOF")
    return bytes(stream)


def create_sample_pdf(path: str | Path, text: str | None = None) -> Path:
    """Write a sample PDF to disk and return the created path."""
    pdf_path = Path(path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(build_sample_pdf_bytes(text=text))
    return pdf_path
