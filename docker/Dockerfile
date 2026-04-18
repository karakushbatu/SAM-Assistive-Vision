# ============================================================
# SAM Assistive Vision — Dockerfile
# ============================================================
# GPU-ready base image for FastSAM + BLIP inference.
# ============================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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
