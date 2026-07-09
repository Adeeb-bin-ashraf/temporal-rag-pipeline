"""Client entry point for the Temporal RAG pipeline."""

from __future__ import annotations

import asyncio
import logging
import sys
from uuid import uuid4

from temporalio.client import Client

from app.config import configure_logging, get_settings
from app.constants import TASK_QUEUE_NAME
from app.workflows import DocumentIndexWorkflow, QuestionAnswerWorkflow

logger = logging.getLogger(__name__)


async def main() -> None:
    """Submit indexing or question-answering workflows to Temporal."""
    configure_logging()
    settings = get_settings()
    client = await Client.connect(settings.temporal_server)

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python client.py index <pdf_path> | ask <question>")

    command = sys.argv[1]
    if command == "index":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: python client.py index <pdf_path>")

        pdf_path = sys.argv[2]
        handle = await client.start_workflow(
            DocumentIndexWorkflow.run,
            pdf_path,
            id=f"index-workflow-{uuid4().hex}",
            task_queue=TASK_QUEUE_NAME,
        )
        result = await handle.result()
        logger.info("Completed indexing workflow: %s", handle.id)
        print(f"Workflow: {handle.id}")
        print(f"Indexed file: {pdf_path}")
        print(f"Result: {result}")
        return

    if command == "ask":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: python client.py ask <question>")

        question = sys.argv[2]
        handle = await client.start_workflow(
            QuestionAnswerWorkflow.run,
            question,
            id=f"ask-workflow-{uuid4().hex}",
            task_queue=TASK_QUEUE_NAME,
        )
        result = await handle.result()
        logger.info("Completed question workflow: %s", handle.id)
        print(f"Question: {question}")
        print(f"Retrieved context: {result.get('retrieved_context', [])}")
        print(f"Final answer: {result.get('answer', '')}")
        return

    raise SystemExit("Usage: python client.py index <pdf_path> | ask <question>")


if __name__ == "__main__":
    asyncio.run(main())
