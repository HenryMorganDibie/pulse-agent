FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default: run the API server
# PORT is injected by Render at runtime; falls back to 8000 for local Docker
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
