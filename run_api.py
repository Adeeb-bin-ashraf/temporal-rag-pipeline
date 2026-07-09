"""Entry point for the web API + frontend server.

Run this alongside the Temporal worker (``python worker.py``):

    python run_api.py

Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import os

import uvicorn

from app.config import configure_logging


def main() -> None:
    configure_logging()
    # Bind all interfaces by default so the server is reachable from outside a
    # container; override with API_HOST=127.0.0.1 for loopback-only locally.
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("app.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
