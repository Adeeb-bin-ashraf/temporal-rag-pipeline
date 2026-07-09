"""FastAPI bridge between the web frontend and the Temporal RAG pipeline.

This layer is intentionally thin: every request is translated into a Temporal
workflow start / query / result call. All heavy lifting still happens inside the
existing activities and workflows, so the HTTP API adds a presentation surface
without duplicating or bypassing the orchestration.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError

from app.config import get_settings
from app.constants import ASK_STAGES, INDEX_STAGES, TASK_QUEUE_NAME, WEB_STAGES
from app.workflows import DocumentIndexWorkflow, QuestionAnswerWorkflow, WebIndexWorkflow

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Temporal RAG Pipeline", version="1.0.0")

_client: Client | None = None


async def get_client() -> Client:
    """Return a cached Temporal client, connecting on first use."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = await Client.connect(settings.temporal_server)
    return _client


# --------------------------------------------------------------------------- #
# Health / metadata
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Report the reachability of every external dependency the pipeline needs."""
    settings = get_settings()
    status: dict[str, Any] = {"temporal": False, "qdrant": False, "ollama": False, "vectors": None}

    try:
        client = await get_client()
        # Issue a real liveness RPC every call so the flag tracks current
        # reachability (a cached client alone would report stale "healthy").
        await client.service_client.check_health()
        status["temporal"] = True
    except Exception as exc:  # noqa: BLE001 - health checks report, never raise
        logger.warning("Temporal health check failed: %s", exc)

    def _qdrant_probe() -> int | None:
        from app.vector_store import VectorStore

        store = VectorStore()
        try:
            return store.count()
        except Exception:  # collection may not exist yet
            store.client.get_collections()
            return 0

    try:
        status["vectors"] = await asyncio.to_thread(_qdrant_probe)
        status["qdrant"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant health check failed: %s", exc)

    def _ollama_probe() -> bool:
        from ollama import Client as OllamaClient

        OllamaClient().list()
        return True

    try:
        status["ollama"] = await asyncio.to_thread(_ollama_probe)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama health check failed: %s", exc)

    status["model"] = settings.ollama_model
    status["embedding_model"] = settings.embedding_model
    status["collection"] = settings.qdrant_collection
    return status


@app.get("/api/pipeline")
async def pipeline() -> dict[str, Any]:
    """Expose the stage definitions so the UI can render the pipeline up front."""
    return {"index": INDEX_STAGES, "web": WEB_STAGES, "ask": ASK_STAGES}


@app.post("/api/reset")
async def reset() -> dict[str, Any]:
    """Drop the vector collection so a demo can start from a clean slate."""

    def _drop() -> None:
        from app.vector_store import VectorStore

        store = VectorStore()
        try:
            store.delete_collection()
        except Exception:  # collection may not exist yet — that's fine
            pass

    await asyncio.to_thread(_drop)
    return {"ok": True, "vectors": 0}


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
@app.post("/api/index/start")
async def index_start(
    file: UploadFile | None = File(default=None),
    use_sample: str | None = Form(default=None),
) -> dict[str, Any]:
    """Persist an uploaded PDF (or use the bundled sample) and start indexing."""
    settings = get_settings()
    documents_dir = settings.documents_path
    documents_dir.mkdir(parents=True, exist_ok=True)

    if file is not None and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
        safe_name = Path(file.filename).name
        target = documents_dir / safe_name
        target.write_bytes(await file.read())
        pdf_path = str(target)
    elif use_sample:
        sample = documents_dir / "sample.pdf"
        if not sample.exists():
            raise HTTPException(status_code=404, detail="No sample.pdf found in documents/.")
        pdf_path = str(sample)
    else:
        raise HTTPException(status_code=400, detail="Provide a PDF file or set use_sample=true.")

    client = await get_client()
    workflow_id = f"index-{uuid4().hex}"
    await client.start_workflow(
        DocumentIndexWorkflow.run,
        pdf_path,
        id=workflow_id,
        task_queue=TASK_QUEUE_NAME,
    )
    logger.info("Started index workflow %s for %s", workflow_id, pdf_path)
    return {"workflow_id": workflow_id, "file": Path(pdf_path).name}


@app.get("/api/index/status/{workflow_id}")
async def index_status(workflow_id: str) -> dict[str, Any]:
    """Return live per-stage progress and, once finished, the workflow result."""
    return await _workflow_status(workflow_id, DocumentIndexWorkflow.get_progress)


# --------------------------------------------------------------------------- #
# Web page indexing (same pipeline as PDFs, sourced from a URL)
# --------------------------------------------------------------------------- #
@app.post("/api/web/start")
async def web_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Start a workflow that scrapes a URL and indexes it like a document."""
    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = f"https://{url}"

    client = await get_client()
    workflow_id = f"web-{uuid4().hex}"
    await client.start_workflow(
        WebIndexWorkflow.run,
        url,
        id=workflow_id,
        task_queue=TASK_QUEUE_NAME,
    )
    logger.info("Started web index workflow %s for %s", workflow_id, url)
    return {"workflow_id": workflow_id, "url": url}


@app.get("/api/web/status/{workflow_id}")
async def web_status(workflow_id: str) -> dict[str, Any]:
    """Return live per-stage progress and, once finished, the workflow result."""
    return await _workflow_status(workflow_id, WebIndexWorkflow.get_progress)


# --------------------------------------------------------------------------- #
# Question answering
# --------------------------------------------------------------------------- #
@app.post("/api/ask/start")
async def ask_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Start a question-answering workflow for the supplied question."""
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question text cannot be empty.")

    client = await get_client()
    workflow_id = f"ask-{uuid4().hex}"
    await client.start_workflow(
        QuestionAnswerWorkflow.run,
        question,
        id=workflow_id,
        task_queue=TASK_QUEUE_NAME,
    )
    logger.info("Started ask workflow %s", workflow_id)
    return {"workflow_id": workflow_id, "question": question}


@app.get("/api/ask/status/{workflow_id}")
async def ask_status(workflow_id: str) -> dict[str, Any]:
    """Return live per-stage progress and, once finished, the answer + context."""
    return await _workflow_status(workflow_id, QuestionAnswerWorkflow.get_progress)


# --------------------------------------------------------------------------- #
# Shared status helper
# --------------------------------------------------------------------------- #
async def _workflow_status(workflow_id: str, query_fn: Any) -> dict[str, Any]:
    client = await get_client()
    handle = client.get_workflow_handle(workflow_id)

    try:
        description = await handle.describe()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Workflow not found: {exc}") from exc

    state = "running"
    result: Any = None
    error: str | None = None
    progress: list[dict[str, Any]] = []

    status = description.status
    if status == WorkflowExecutionStatus.COMPLETED:
        state = "completed"
        result = await handle.result()
    elif status in (
        WorkflowExecutionStatus.FAILED,
        WorkflowExecutionStatus.TERMINATED,
        WorkflowExecutionStatus.TIMED_OUT,
        WorkflowExecutionStatus.CANCELED,
    ):
        state = "failed"
        try:
            await handle.result()
        except WorkflowFailureError as exc:
            error = str(exc.cause or exc)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    # The progress query works while running and after completion (within
    # retention). It can briefly race a just-started workflow, so failures here
    # are non-fatal.
    try:
        progress = await handle.query(query_fn)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Progress query not ready for %s: %s", workflow_id, exc)

    return {"workflow_id": workflow_id, "state": state, "progress": progress, "result": result, "error": error}


# --------------------------------------------------------------------------- #
# Static frontend (mounted last so /api/* routes take precedence)
# --------------------------------------------------------------------------- #
@app.get("/")
async def index_page() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"detail": "Frontend not built. See frontend/index.html."}, status_code=404)
    return FileResponse(index_file)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
