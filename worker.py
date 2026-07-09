"""Temporal worker entry point for the RAG pipeline."""

from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

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
from app.config import configure_logging, get_settings
from app.constants import TASK_QUEUE_NAME
from app.workflows import DocumentIndexWorkflow, QuestionAnswerWorkflow, WebIndexWorkflow

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """Connect to Temporal and run the worker loop."""
    configure_logging()
    settings = get_settings()
    logger.info("Starting Temporal worker for queue %s", TASK_QUEUE_NAME)
    client = await Client.connect(settings.temporal_server)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE_NAME,
        workflows=[DocumentIndexWorkflow, WebIndexWorkflow, QuestionAnswerWorkflow],
        activities=[
            read_pdf_activity,
            read_url_activity,
            split_document_activity,
            generate_embeddings_activity,
            store_vectors_activity,
            generate_query_embedding_activity,
            retrieve_chunks_activity,
            rerank_chunks_activity,
            generate_answer_activity,
        ],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
