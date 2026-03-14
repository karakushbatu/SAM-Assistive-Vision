# Tech Context

## Python Environment
- Python 3.11+ required
- Virtual environment: `python -m venv venv`
- Activation (Windows): `venv\Scripts\activate`
- Activation (Linux/Mac): `source venv/bin/activate`

## Key Dependencies
| Package | Purpose |
|---|---|
| fastapi | HTTP + WebSocket framework |
| uvicorn[standard] | ASGI server (runs FastAPI) |
| python-multipart | Form/file upload support |
| pydantic-settings | Env-var based config |
| pillow | Image decoding (JPEG/PNG bytes → PIL Image) |
| numpy | Array operations on image data |
| httpx | Async HTTP client (used for Ollama API calls in Phase 4) |

## Docker Setup
- Base image: `python:3.11-slim`
- For GPU: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- docker-compose uses `deploy.resources.reservations.devices` for GPU passthrough
- Requires: NVIDIA Container Toolkit on host (`nvidia-container-runtime`)

## Ollama Setup (Phase 4)
- Ollama runs as a separate container or local process on port 11434
- API endpoint: `http://ollama:11434/api/generate`
- Model to pull: `llama3:8b`
- Command to pull model: `docker exec -it ollama ollama pull llama3:8b`

## Local Dev Commands
```bash
# Create venv
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dev server (auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run via Docker Compose
docker compose up --build

# Run with GPU (Phase 2+)
docker compose up --build  # GPU config already in docker-compose.yml
```
