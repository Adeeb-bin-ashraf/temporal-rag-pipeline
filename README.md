# Temporal RAG Pipeline

A full-stack **Retrieval-Augmented Generation** system where every step of the pipeline —
**PDF and web-page** ingestion, chunking, embedding, vector storage, retrieval, and grounded
answer generation — is orchestrated as a **Temporal workflow**, and visualized **live** in a web UI.

The web app doesn't fake progress: it streams each stage's real status straight from the
running workflow using Temporal **workflow queries**, so you watch `Read PDF → Split → Embed → Store`
(and `Embed Query → Retrieve → Generate`) complete in real time, annotated with the actual
page/chunk/vector counts and similarity scores.

```
┌───────────────────────────────────────────────────────────────────────┐
│  Browser (single-page dashboard)                                        │
│    upload PDF / ask question  ──►  poll live per-stage progress         │
└───────────────┬─────────────────────────────────▲─────────────────────┘
                │ HTTP (JSON / multipart)          │ @workflow.query progress
                ▼                                   │
┌───────────────────────────────────────────────────────────────────────┐
│  FastAPI bridge (app/api.py)  ── starts / queries / awaits workflows    │
└───────────────┬─────────────────────────────────▲─────────────────────┘
                │ Temporal client                  │
                ▼                                   │
┌───────────────────────────────────────────────────────────────────────┐
│  Temporal worker (worker.py) ── DocumentIndexWorkflow / QAWorkflow      │
│    activities: read_pdf · split · embed · store · retrieve · generate   │
└──────┬──────────────┬───────────────┬───────────────────┬──────────────┘
       ▼              ▼               ▼                   ▼
  PyPDF2 loader   Sentence-       Qdrant vector      Ollama LLM
                  Transformers    database           (qwen2.5:3b)
```

## Highlights

- **Temporal-orchestrated pipeline** — each responsibility is an isolated activity with its
  own timeout and retry policy, so a transient Qdrant/Ollama hiccup retries a single step
  instead of reprocessing the whole document.
- **Live pipeline visualization** — the workflows record per-stage progress and expose it via
  a `@workflow.query`; the frontend polls it, so the UI reflects real workflow state, not a
  canned animation.
- **Two ingestion sources, one pipeline** — index a PDF *or* scrape a web page by URL; both feed
  the identical split → embed → store path (`WebIndexWorkflow` reuses the document activities).
- **Retrieve-then-rerank** — dense search fetches a wide candidate set, then a cross-encoder
  reranker (`ms-marco-MiniLM`) re-scores each (question, chunk) pair and keeps the best few, so
  the LLM sees genuinely relevant context rather than just embedding-nearest chunks.
- **Sentence-aware chunking + idempotent indexing** — chunks break on sentence boundaries with
  character overlap; deterministic point IDs mean re-indexing the same document upserts instead
  of accumulating duplicate vectors.
- **One-command startup** — `docker compose up` brings up Temporal, Qdrant, the worker, and the
  API together; open the browser and go.
- **Markdown-formatted answers** — the LLM is prompted for clean Markdown, rendered client-side
  (headings, lists, bold, code) so responses read as formatted prose, not a wall of text.
- **Clean separation of concerns** — reusable business logic (`app/`) is fully decoupled from
  orchestration (`workflows.py` / `activities.py`) and from presentation (`api.py` / `frontend/`).
- **Grounded answers** — the LLM is prompted to answer *only* from retrieved context, and the
  UI surfaces the exact chunks, sources, and cosine-similarity scores behind every answer.
- **Zero-build frontend** — a self-contained HTML/CSS/JS dashboard served by FastAPI; no Node
  toolchain required.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest [Indexing workflow]
        PDF[PDF] --> L[Read PDF] --> S[Split Text] --> E[Embed Chunks] --> V[(Qdrant)]
    end
    subgraph Query [Question workflow]
        Q[Question] --> QE[Embed Query] --> R[Retrieve] --> RR[Rerank] --> G[Generate Answer]
        V --> R
        G --> A[Grounded Answer]
    end
    UI[Web UI] -->|HTTP| API[FastAPI]
    API -->|Temporal client| WF[Temporal Worker]
    WF -->|@workflow.query| API
```

## Tech Stack

| Layer          | Technology                                         |
|----------------|----------------------------------------------------|
| Orchestration  | Temporal Python SDK                                |
| Embeddings     | Hugging Face Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Reranking      | Cross-encoder (`ms-marco-MiniLM-L-6-v2`)           |
| Vector store   | Qdrant (cosine similarity)                         |
| LLM            | Ollama (`qwen2.5:3b`)                              |
| PDF parsing    | PyPDF2                                             |
| Web scraping   | httpx + BeautifulSoup                              |
| Web API        | FastAPI + Uvicorn                                  |
| Frontend       | Vanilla HTML / CSS / JavaScript (no build step)    |

## Quickstart (Docker) — recommended

The whole system (Temporal + Qdrant + worker + API) comes up with one command.

```bash
# Ollama runs on the host (GPU-accelerated). Install it once and pull the model:
ollama pull qwen2.5:3b

docker compose up --build
```

Then open **http://localhost:8000** (app) and **http://localhost:8233** (Temporal Web UI).
The containerized app reaches your host Ollama automatically via `host.docker.internal`.

> First build takes a few minutes (it installs PyTorch). Subsequent runs are cached.

Prefer a fully self-contained run with **Ollama in a container too** (no host install; CPU-only
unless a GPU is configured)? Add `OLLAMA_HOST=http://ollama:11434` to a `.env` file and run:

```bash
docker compose --profile ollama up --build   # first run pulls the ~2GB model
```

---

## Manual setup (without Docker)

### Prerequisites

- Python 3.13
- A running **Temporal** dev server
- A running **Qdrant** instance
- A running **Ollama** instance with the `qwen2.5:3b` model pulled

### Start the services

```bash
# Temporal dev server (Temporal CLI)
temporal server start-dev            # serves gRPC on localhost:7233, UI on :8233

# Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Ollama
ollama serve
ollama pull qwen2.5:3b
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env                 # optional; defaults already match .env.example
```

## Running

Open **three** terminals (all four external services above must be up first):

```bash
# 1. Temporal worker — registers the workflows + activities
python worker.py

# 2. Web API + frontend
python run_api.py                    # http://localhost:8000

# 3. (optional) CLI instead of / alongside the UI
python client.py index documents/sample.pdf
python client.py ask "What is this document about?"
```

Then open **http://localhost:8000** and:

1. In **Add knowledge**, choose **PDF** (drop a file or use the bundled sample) or **Web URL**
   (paste a link) and index it — watch the four ingestion stages complete in a live stepper.
2. In **Ask a question**, type a question — watch embed → retrieve → **rerank** → generate run,
   then read the Markdown-formatted, grounded answer with a collapsible list of source chunks
   and scores.

`POST /api/reset` clears the Qdrant collection for a fresh demo.

## HTTP API

| Method | Endpoint                     | Purpose                                            |
|--------|------------------------------|----------------------------------------------------|
| GET    | `/api/health`                | Reachability of Temporal / Qdrant / Ollama + vector count |
| GET    | `/api/pipeline`              | Stage definitions used to render the UI            |
| POST   | `/api/index/start`           | Upload a PDF (or `use_sample=true`); starts indexing workflow |
| GET    | `/api/index/status/{id}`     | Live per-stage progress + final result             |
| POST   | `/api/web/start`             | `{ "url": "..." }`; scrapes + indexes a web page   |
| GET    | `/api/web/status/{id}`       | Live per-stage progress + final result             |
| POST   | `/api/ask/start`             | `{ "question": "..." }`; starts Q&A workflow       |
| GET    | `/api/ask/status/{id}`       | Live per-stage progress + answer, context, scores  |
| POST   | `/api/reset`                 | Drop the vector collection                         |

## Project Structure

```
app/
  activities.py     # Temporal activities — one per pipeline responsibility
  workflows.py      # DocumentIndex / WebIndex / QuestionAnswer workflows + @workflow.query progress
  api.py            # FastAPI bridge: HTTP  ->  Temporal start/query/result
  rag_service.py    # Reusable RAG orchestration (usable outside Temporal)
  embeddings.py     # Sentence-Transformers wrapper (cached model)
  reranker.py       # Cross-encoder reranking of retrieved candidates
  pdf_loader.py     # PyPDF2 loading + text normalization
  web_loader.py     # httpx + BeautifulSoup web-page text extraction
  text_splitter.py  # Sentence-aware chunking with character overlap
  vector_store.py   # Qdrant management + search + deterministic (dedup) IDs
  config.py         # Environment-driven settings
  constants.py      # Task queue name, defaults, pipeline stage definitions
frontend/
  index.html        # Dashboard markup
  styles.css        # Theme + layout
  app.js            # Fetch/poll logic, live pipeline rendering, Markdown renderer
worker.py           # Temporal worker entry point
client.py           # CLI entry point (index / ask)
run_api.py          # Web server entry point
Dockerfile          # App image (worker + api)
docker-compose.yml  # One-command full stack
documents/          # Input PDFs (sample.pdf included)
tests/              # Lightweight end-to-end integration script
```

## How a request flows

1. The browser `POST`s to `/api/index/start` (or `/api/ask/start`); FastAPI starts the
   corresponding Temporal workflow and returns its `workflow_id` immediately.
2. As each activity completes, the workflow updates an in-memory progress list.
3. The browser polls `/api/{index,ask}/status/{id}`, which issues a Temporal **query** to read
   that live progress and, once the workflow status is `COMPLETED`, returns the final result.
4. The UI renders each stage's state and the final answer + retrieved context.

## Design decisions

- **Why Temporal for a RAG pipeline?** Indexing and answering are multi-step processes where any
  step can fail transiently (model cold-start, Qdrant unavailable, Ollama busy). Modeling each
  step as an activity gives per-step timeouts, retries, and a durable execution history for free,
  and exposing progress via `@workflow.query` makes the pipeline observable in real time. The
  honest trade-off: Temporal is heavier than a plain script for a single-machine demo — it's used
  here to demonstrate durable orchestration, and the same activities remain callable outside
  Temporal (`RAGService`) so the business logic isn't locked to it.
- **Why retrieve-then-rerank?** Dense vector search is fast but ranks only by embedding proximity.
  A cross-encoder that reads (question, chunk) together is far more accurate but too slow to run
  over the whole corpus — so we retrieve a wide candidate set cheaply, then rerank a handful.
- **Why deterministic point IDs?** Re-indexing the same document should be idempotent; hashing
  (source + text) into the ID means repeats upsert instead of piling up duplicate vectors.

## Extensibility

The design is intentionally modular — batch indexing, multi-PDF ingestion, metadata-filtered
retrieval, streaming LLM responses, and hybrid (dense + sparse) search can all be added without
rewriting the core pipeline.
