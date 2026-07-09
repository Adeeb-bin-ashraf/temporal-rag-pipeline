# Image for both the Temporal worker and the FastAPI web server.
# The same image runs two commands (see docker-compose.yml): `python worker.py`
# and `python run_api.py`.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

WORKDIR /app

# libgomp1 is needed by torch; curl is handy for container healthchecks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first from the CPU wheel index so the huge default
# CUDA build isn't pulled in, then the rest of the requirements (which see torch
# as already satisfied). This keeps the image a few GB smaller.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding + reranker models into the image (HF_HOME=/models)
# so the very first index/ask doesn't fetch them mid-activity — a fresh
# `docker compose up` then works on the first try.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

EXPOSE 8000

CMD ["python", "run_api.py"]
