# KFL Backend GPU v5 - Backfill Pipeline
# ======================================
# Docker container voor GPU-accelerated indicator & signal backfill
# Met CuPy (GPU) en Numba (CPU) ondersteuning

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Metadata
LABEL maintainer="KlineFuturesLab"
LABEL version="5.0.0"
LABEL description="GPU-accelerated backfill pipeline for indicators and signals"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    wget \
    git \
    postgresql-client \
    libpq-dev \
    pkg-config \
    vim \
    nano \
    htop \
    tree \
    cmake \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy application code
COPY . .

# Copy launcher script
COPY start /usr/local/bin/start
RUN chmod +x /usr/local/bin/start

# Create directories
RUN mkdir -p /app/_log && chmod 755 /app/_log

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Non-root user for security
RUN useradd -m kfluser && chown -R kfluser:kfluser /app

# REASON: Entrypoint draait als root om database parameters in te stellen
# USER kfluser

# Expose port (if needed)
EXPOSE 8080

# Entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - interactive bash
CMD ["/bin/bash"]
