"""Create a sample PDF for local development and testing."""

from __future__ import annotations

from pathlib import Path

from app.sample_pdf import create_sample_pdf


def main() -> None:
    """Write a sample PDF into the documents directory and print its path."""
    pdf_path = create_sample_pdf(Path("documents/sample.pdf"))
    print(pdf_path)


if __name__ == "__main__":
    main()
