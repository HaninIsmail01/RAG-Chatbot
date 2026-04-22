# Base image
# Python 3.11 slim keeps the image lean 
#Slim avoids the full Debian image (~900MB).
FROM python:3.11-slim

#  Build arguments 
ARG PORT=8000

# Environment variables 
# PYTHONDONTWRITEBYTECODE: prevents .pyc files inside the container
# PYTHONUNBUFFERED: ensures logs appear immediately (critical for Chainlit)
# PYTHONPATH: makes config/ and src/ importable from anywhere
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=${PORT}

#  Working directory 
WORKDIR /app

# System dependencies
# gcc and g++ are required to compile some Python packages (e.g. onnxruntime)
# curl is useful for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements first — Docker layer caching means this layer
# is only rebuilt when requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source 
# Copied after pip install to maximize cache reuse
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Pre-download FastEmbed model weights 
# FastEmbed downloads model weights on first use. This is triggered
# during the build so the container starts instantly without a
# download delay. The weights are cached at .fastembed_cache/
RUN python -c "\
from fastembed import TextEmbedding; \
model = TextEmbedding('BAAI/bge-base-en-v1.5'); \
print('FastEmbed model cached successfully')"

#  Pre-download reranker model weights 
# Same rationale — sentence-transformers downloads on first use
RUN python -c "\
from sentence_transformers import CrossEncoder; \
model = CrossEncoder('BAAI/bge-reranker-base'); \
print('Reranker model cached successfully')"`

# Expose port
EXPOSE ${PORT}

# Healthcheck 
# Confirms the app is responding before Docker marks it healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Entrypoint 
# Run Chainlit directly — no shell wrapper needed
CMD ["sh", "-c", "chainlit run src/ui/app.py --port ${PORT} --host 0.0.0.0"]