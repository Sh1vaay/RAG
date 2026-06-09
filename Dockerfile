# ==============================================================================
# Build Stage
# ==============================================================================
FROM python:3.12-slim-bookworm AS builder

# Prevent python from writing pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml requirements.txt uv.lock ./

# Install python dependencies system-wide in builder stage
RUN uv pip install --system --no-cache -r requirements.txt

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy system-wide packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create required runtime directories and set permissions
RUN mkdir -p /app/documents /app/faiss_db && \
    useradd -u 10001 -U appuser && \
    chown -R appuser:appuser /app

# Run as a non-root user (Principle of Least Privilege)
USER appuser

EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
