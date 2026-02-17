# Build stage: Use official Python image as base
FROM python:3.12-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md ./
COPY histoslice /app/histoslice

# Install dependencies using uv pip (more robust for network issues)
RUN uv venv && \
    uv pip install --no-cache -e .

# Runtime stage: Create minimal runtime image
FROM python:3.12-slim

# Install system dependencies for OpenCV (headless)
# Note: opencv-python-headless requires minimal system libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install UV in runtime image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY histoslice /app/histoslice
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md ./

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Create directory for input/output
RUN mkdir -p /data/input /data/output

# Set default working directory for data
WORKDIR /data

# Default command runs the CLI help
ENTRYPOINT ["histoslice"]
CMD ["--help"]
