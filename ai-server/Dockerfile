# ────────────────────────────────────────────────────────────────────────────
# Workflow360 AI Server — HuggingFace Spaces Dockerfile
# ────────────────────────────────────────────────────────────────────────────
# Free tier: 16 GB RAM, 2 vCPUs, 50 GB storage, no spin-down.
# Fits all 3 AI models (FLAN-T5, SBERT+sklearn, XGBoost) comfortably.
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user id 1000 with read/write to /app
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements first for layer caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download SBERT model to image (avoids cold start delay)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY --chown=user:user . .

USER user
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV MODEL_DIR=/app/model_weights
ENV DEVICE=cpu

# HF Spaces expects port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
