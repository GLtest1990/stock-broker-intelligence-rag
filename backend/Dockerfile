# Slim Python base image
FROM python:3.11-slim

# Show Python output immediately in `docker logs` and skip .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

# Set the working directory inside the container
WORKDIR /app

# OpenMP runtime needed by FAISS and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (much smaller than the default CUDA build); no GPU is needed for MiniLM
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies (requirements.txt is copied first so this layer is cached between builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model once at build time, then run fully offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

# Copy the API code and the pre-built FAISS vector database (vectordb/)
COPY . .

EXPOSE 7860

# `docker ps` shows (healthy) / (unhealthy). python is used because the slim image has no curl.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=4)"

# Start the API with Gunicorn
# - `-w 1`: one worker process (each worker loads its own copy of the MiniLM model and vector DB)
# - `--threads 4`: the worker handles several requests concurrently
# - `--timeout 300`: also applies while the worker boots (PyTorch + MiniLM + FAISS load), so it must be generous or
#   gunicorn kills the booting worker in a loop and the port never answers
# - `--access-logfile/--error-logfile -`: request and worker-crash messages appear in `docker logs backend`
# - `app:rag_api`: the Flask instance named `rag_api` defined in app.py
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "-b", "0.0.0.0:7860", "app:rag_api"]
