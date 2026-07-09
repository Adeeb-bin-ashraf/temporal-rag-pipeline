"""Temporal workflows for the RAG pipeline.

The workflows stay thin and delegate the real work to activities so the
business logic remains reusable outside Temporal as well. Each workflow also
records per-stage progress and exposes it through a ``@workflow.query`` so a
frontend can visualize the pipeline live while it executes.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.activities import (
        generate_answer_activity,
        generate_embeddings_activity,
        generate_query_embedding_activity,
        read_pdf_activity,
        read_url_activity,
        rerank_chunks_activity,
        retrieve_chunks_activity,
        split_document_activity,
        store_vectors_activity,
    )
    from app.constants import (
        ASK_STAGES,
        DEFAULT_QUERY_LIMIT,
        INDEX_STAGES,
        RERANK_CANDIDATE_LIMIT,
        WEB_STAGES,
    )
    from app.pdf_loader import DocumentPage
    from app.text_splitter import DocumentChunk
    from app.vector_store import SearchResult

logger = logging.getLogger(__name__)


def _retry_policy(
    maximum_attempts: int = 3,
    maximum_interval: int = 20,
) -> RetryPolicy:
    """Create a consistent retry policy for workflow activities."""

    return RetryPolicy(
        maximum_attempts=maximum_attempts,
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=maximum_interval),
        non_retryable_error_types=["ValueError"],
    )


def _timeout(seconds: int) -> timedelta:
    """Convert a duration in seconds into the timedelta type expected by Temporal."""
    return timedelta(seconds=seconds)


class _ProgressMixin:
    """Shared per-stage progress tracking exposed to clients via a query.

    Progress is a plain list of dicts so it serializes cleanly through
    Temporal's default JSON data converter and is trivial to render in a UI.
    """

    def _init_progress(self, stages: list[dict[str, str]]) -> None:
        self._progress: list[dict[str, Any]] = [
            {"key": stage["key"], "label": stage["label"], "status": "pending", "detail": ""}
            for stage in stages
        ]

    def _set_stage(self, key: str, status: str, detail: str = "") -> None:
        for entry in self._progress:
            if entry["key"] == key:
                entry["status"] = status
                entry["detail"] = detail
                break

    def _fail_current_stage(self) -> None:
        """Mark the first not-yet-completed stage as errored."""
        for entry in self._progress:
            if entry["status"] in ("running", "pending"):
                entry["status"] = "error"
                break

    @workflow.query
    def get_progress(self) -> list[dict[str, Any]]:
        """Return the current per-stage progress for live visualization."""
        return self._progress


@workflow.defn
class DocumentIndexWorkflow(_ProgressMixin):
    """Index a PDF document by orchestrating the existing RAG modules."""

    def __init__(self) -> None:
        self._init_progress(INDEX_STAGES)

    @workflow.run
    async def run(self, pdf_path: str) -> dict[str, Any]:
        logger.info("Workflow started: DocumentIndexWorkflow for %s", pdf_path)

        try:
            self._set_stage("read_pdf", "running")
            pages = await workflow.execute_activity(
                read_pdf_activity,
                pdf_path,
                start_to_close_timeout=_timeout(30),
                retry_policy=_retry_policy(maximum_interval=10),
            )
            self._set_stage("read_pdf", "done", f"{len(pages)} page(s)")

            self._set_stage("split", "running")
            chunks = await workflow.execute_activity(
                split_document_activity,
                args=[pages, Path(pdf_path).name],
                start_to_close_timeout=_timeout(30),
                retry_policy=_retry_policy(maximum_interval=10),
            )
            self._set_stage("split", "done", f"{len(chunks)} chunk(s)")

            self._set_stage("embed", "running")
            embeddings = await workflow.execute_activity(
                generate_embeddings_activity,
                chunks,
                start_to_close_timeout=_timeout(120),
                retry_policy=_retry_policy(),
            )
            dimension = len(embeddings[0]) if embeddings else 0
            self._set_stage("embed", "done", f"{len(embeddings)} vector(s), dim={dimension}")

            self._set_stage("store", "running")
            stored = await workflow.execute_activity(
                store_vectors_activity,
                args=[chunks, embeddings],
                start_to_close_timeout=_timeout(60),
                retry_policy=_retry_policy(),
            )
            inserted = stored.get("inserted", 0)
            self._set_stage("store", "done", f"{inserted} inserted")
        except Exception:
            self._fail_current_stage()
            raise

        logger.info("Workflow completed: DocumentIndexWorkflow for %s", pdf_path)
        return {
            "file": pdf_path,
            "pages": len(pages),
            "chunks": len(chunks),
            "inserted": stored.get("inserted", 0),
        }


@workflow.defn
class WebIndexWorkflow(_ProgressMixin):
    """Index a web page by fetching it and reusing the shared RAG modules."""

    def __init__(self) -> None:
        self._init_progress(WEB_STAGES)

    @workflow.run
    async def run(self, url: str) -> dict[str, Any]:
        logger.info("Workflow started: WebIndexWorkflow for %s", url)

        try:
            self._set_stage("fetch_url", "running")
            pages = await workflow.execute_activity(
                read_url_activity,
                url,
                start_to_close_timeout=_timeout(60),
                retry_policy=_retry_policy(maximum_interval=10),
            )
            self._set_stage("fetch_url", "done", f"{len(pages)} page(s)")

            self._set_stage("split", "running")
            chunks = await workflow.execute_activity(
                split_document_activity,
                args=[pages, url],
                start_to_close_timeout=_timeout(30),
                retry_policy=_retry_policy(maximum_interval=10),
            )
            self._set_stage("split", "done", f"{len(chunks)} chunk(s)")

            self._set_stage("embed", "running")
            embeddings = await workflow.execute_activity(
                generate_embeddings_activity,
                chunks,
                start_to_close_timeout=_timeout(120),
                retry_policy=_retry_policy(),
            )
            dimension = len(embeddings[0]) if embeddings else 0
            self._set_stage("embed", "done", f"{len(embeddings)} vector(s), dim={dimension}")

            self._set_stage("store", "running")
            stored = await workflow.execute_activity(
                store_vectors_activity,
                args=[chunks, embeddings],
                start_to_close_timeout=_timeout(60),
                retry_policy=_retry_policy(),
            )
            inserted = stored.get("inserted", 0)
            self._set_stage("store", "done", f"{inserted} inserted")
        except Exception:
            self._fail_current_stage()
            raise

        logger.info("Workflow completed: WebIndexWorkflow for %s", url)
        return {
            "url": url,
            "pages": len(pages),
            "chunks": len(chunks),
            "inserted": stored.get("inserted", 0),
        }


@workflow.defn
class QuestionAnswerWorkflow(_ProgressMixin):
    """Answer a user question by orchestrating retrieval and generation."""

    def __init__(self) -> None:
        self._init_progress(ASK_STAGES)

    @workflow.run
    async def run(self, question: str) -> dict[str, Any]:
        logger.info("Workflow started: QuestionAnswerWorkflow for %s", question)

        try:
            self._set_stage("embed_query", "running")
            query_vector = await workflow.execute_activity(
                generate_query_embedding_activity,
                question,
                start_to_close_timeout=_timeout(120),
                retry_policy=_retry_policy(),
            )
            self._set_stage("embed_query", "done", f"dim={len(query_vector)}")

            self._set_stage("retrieve", "running")
            candidates = await workflow.execute_activity(
                retrieve_chunks_activity,
                args=[query_vector, RERANK_CANDIDATE_LIMIT],
                start_to_close_timeout=_timeout(60),
                retry_policy=_retry_policy(),
            )
            self._set_stage("retrieve", "done", f"{len(candidates)} candidate(s)")

            self._set_stage("rerank", "running")
            search_results = await workflow.execute_activity(
                rerank_chunks_activity,
                args=[question, candidates, DEFAULT_QUERY_LIMIT],
                start_to_close_timeout=_timeout(120),  # absorbs first-time reranker model load
                retry_policy=_retry_policy(),
            )
            self._set_stage("rerank", "done", f"top {len(search_results)}")

            self._set_stage("generate", "running")
            answer_result = await workflow.execute_activity(
                generate_answer_activity,
                args=[question, search_results],
                start_to_close_timeout=_timeout(120),
                retry_policy=_retry_policy(),
            )
            self._set_stage("generate", "done", "answer ready")
        except Exception:
            self._fail_current_stage()
            raise

        logger.info("Workflow completed: QuestionAnswerWorkflow for %s", question)
        return {
            "question": question,
            "retrieved_context": answer_result.get("context_chunks", []),
            "scores": [round(item.score, 4) for item in search_results],
            "sources": [item.metadata for item in search_results],
            "answer": answer_result.get("answer", ""),
        }
