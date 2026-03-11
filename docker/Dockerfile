# ============================================================
# SAM Assistive Vision — Dockerfile
# ============================================================
# Phase 1-3: CPU-only base image (slim + fast builds)
# Phase 4+:  Switch FROM line to nvidia/cuda image (see comment below)
# ============================================================

# --- Phase 1-3 (current): CPU base ---
FROM python:3.11-slim

# --- Phase 4+ (GPU): Uncomment this and comment the line above ---
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- When using GPU base image, also install Python ---
# RUN apt-get update && apt-get install -y python3.11 python3-pip \
#     && ln -s /usr/bin/python3.11 /usr/bin/python \
#     && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Install Python dependencies ----
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application source ----
COPY . .

# ---- Expose FastAPI port ----
EXPOSE 8000

# ---- Start server ----
# --workers 1: single worker is correct for WebSocket apps in development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
