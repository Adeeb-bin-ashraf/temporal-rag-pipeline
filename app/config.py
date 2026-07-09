"""Application configuration for the Temporal RAG pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = BASE_DIR / ".env"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration values loaded from environment variables."""

    temporal_server: str = "localhost:7233"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"
    ollama_model: str = "qwen2.5:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    documents_directory: str = "documents"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration values from the environment and optional .env file."""
        load_dotenv(ENV_FILE, override=False)
        return cls(
            temporal_server=_get_env("TEMPORAL_SERVER", cls.__dataclass_fields__["temporal_server"].default),
            qdrant_host=_get_env("QDRANT_HOST", cls.__dataclass_fields__["qdrant_host"].default),
            qdrant_port=_get_int_env("QDRANT_PORT", cls.__dataclass_fields__["qdrant_port"].default),
            qdrant_collection=_get_env("QDRANT_COLLECTION", cls.__dataclass_fields__["qdrant_collection"].default),
            ollama_model=_get_env("OLLAMA_MODEL", cls.__dataclass_fields__["ollama_model"].default),
            embedding_model=_get_env("EMBEDDING_MODEL", cls.__dataclass_fields__["embedding_model"].default),
            documents_directory=_get_env("DOCUMENTS_DIRECTORY", cls.__dataclass_fields__["documents_directory"].default),
            log_level=_get_env("LOG_LEVEL", cls.__dataclass_fields__["log_level"].default),
        )

    @property
    def documents_path(self) -> Path:
        """Return the configured documents directory as a Path object."""
        return BASE_DIR / self.documents_directory


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int_env(name: str, default: int) -> int:
    value: str = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging using the provided application settings."""
    resolved: Settings = settings or get_settings()
    log_level: int = getattr(logging, resolved.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_settings() -> Settings:
    """Return the globally configured application settings."""
    return settings


settings: Settings = Settings.from_env()
configure_logging(settings)

__all__ = ["BASE_DIR", "ENV_FILE", "Settings", "configure_logging", "get_settings", "settings"]
