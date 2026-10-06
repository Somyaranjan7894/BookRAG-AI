# ==============================================================================
# BookRAG AI - Backend Dockerfile (Phase 19 Production Readiness)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies (build-essential, curl, libgomp for FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and migrations
COPY backend /app/backend
COPY evaluation /app/evaluation

# Set PYTHONPATH so app and evaluation packages are resolvable
ENV PYTHONPATH="/app/backend:/app"

# Expose FastAPI port
EXPOSE 8000

# Default command: launch uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
