"""Shared constants for the Temporal RAG pipeline."""

from __future__ import annotations

TASK_QUEUE_NAME = "rag-task-queue"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

# Retrieval fans out to a wider candidate set, then the cross-encoder reranker
# narrows it to the most relevant few that are actually sent to the LLM.
DEFAULT_QUERY_LIMIT = 4
RERANK_CANDIDATE_LIMIT = 20

# Pipeline stage definitions shared by the workflows (for live progress
# tracking) and the frontend (for rendering the pipeline visualization).
INDEX_STAGES = [
    {"key": "read_pdf", "label": "Read PDF"},
    {"key": "split", "label": "Split Text"},
    {"key": "embed", "label": "Embed Chunks"},
    {"key": "store", "label": "Store Vectors"},
]

WEB_STAGES = [
    {"key": "fetch_url", "label": "Fetch URL"},
    {"key": "split", "label": "Split Text"},
    {"key": "embed", "label": "Embed Chunks"},
    {"key": "store", "label": "Store Vectors"},
]

ASK_STAGES = [
    {"key": "embed_query", "label": "Embed Query"},
    {"key": "retrieve", "label": "Retrieve Chunks"},
    {"key": "rerank", "label": "Rerank"},
    {"key": "generate", "label": "Generate Answer"},
]
