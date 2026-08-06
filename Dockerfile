# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install system dependencies
# build-essential is often needed for compiling C-extensions for data science/vector libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazingly fast dependency installation
RUN pip install uv

# Set up the application directory
WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml ./

# Install the project dependencies using uv
RUN uv pip install --system .

# Copy the rest of the application code
COPY backend/ backend/

# Ensure workspace directory exists and has correct permissions
# (A persistent volume should be mounted here in production)
RUN mkdir -p /app/workspaces && chmod 777 /app/workspaces

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser


# Start the FastAPI server on the dynamically assigned PORT (defaulting to 8000)
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
