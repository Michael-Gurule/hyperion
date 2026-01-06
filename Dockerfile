# =============================================================================
# HYPERION Multi-Agent RL Docker Image
# Multi-stage build for optimized image size
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU version (smaller image, works everywhere)
# For GPU support, replace with: torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy application code
COPY src/ ./src/
COPY config.yaml ./config.yaml

# Create directories for mounted volumes
RUN mkdir -p checkpoints outputs logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hyperion && \
    chown -R hyperion:hyperion /app
USER hyperion

# Expose ports for services
# 8501: Streamlit Dashboard
# 8000: FastAPI (future)
EXPOSE 8501 8000

# Default command: Run dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
