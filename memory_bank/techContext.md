# Tech Context

## Python Environment
- Python 3.14, virtual environment: `python -m venv venv`
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
| httpx | Async HTTP client (Ollama API calls) |
| torch 2.11.0+cu128 | CUDA tensor operations (BLIP + SAM inference) |
| transformers 5.3.0 | BLIP model (BlipProcessor, BlipForConditionalGeneration) |
| accelerate 1.13.0 | HuggingFace model optimization |
| ultralytics | FastSAM-s segmentation model |
| faster-whisper | Optimized Whisper STT (4-10x faster, CTranslate2 backend) |
| sounddevice | Microphone recording (tests/mic_test.py demo only) |
| edge-tts | Edge TTS async client (EmelNeural Turkish voice) |

## Hardware
- GPU: RTX 5060 Ti 16GB VRAM
- CUDA: 12.8 (torch cu128)
- BLIP usage: ~4GB VRAM
- FastSAM usage: ~500MB VRAM
- Ollama: runs separately (qwen2.5:7b = 4.7GB VRAM or CPU RAM)

## Model Files
- BLIP: cached from HuggingFace — `Salesforce/blip-image-captioning-large`
- FastSAM: local file — `FastSAM-s.pt` (ultralytics format)
- faster-whisper: auto-downloaded on first use — `small` model (switched from `tiny` for better Turkish accuracy)
- Ollama models: managed by Ollama app — `qwen2.5:7b` (4.7GB), `llama3.2:3b` (1.9GB)

## Ollama Setup
- Ollama runs as Windows desktop app on port 11434
- API endpoint: `http://localhost:11434/api/generate`
- Active model: `qwen2.5:7b` — set via `OLLAMA_MODEL` in .env
- 2-call architecture per frame:
  1. OCR check: `temperature=0.0, num_predict=3` → "yes" or "no"
  2. Description: `temperature=0.2, num_predict=100` → Turkish 1-2 sentences
- `num_ctx=512` — KV cache reduced from default 4096; prompts use ~300 tokens, 8x memory savings
- Flash Attention: `OLLAMA_FLASH_ATTENTION=1` — must be set in host env before Ollama starts
- Connect timeout: 3s (fast-fail), read timeout: 30s (configurable)
- Model swap: change `OLLAMA_MODEL` in .env — no code change needed

## qwen2.5:3b vs qwen2.5:7b Comparison
To test 3b model (approx 2x faster, Turkish quality unknown):
1. `.env` → `OLLAMA_MODEL=qwen2.5:3b`
2. `ollama pull qwen2.5:3b` (if not already downloaded)
3. `python tests/demo.py` — latency and description printed to console
4. Compare: latency (Ollama line) and Turkish output quality
5. If quality acceptable: keep 3b. If degraded: revert to `OLLAMA_MODEL=qwen2.5:7b`
Note: demo.py header now shows active model name + num_ctx for easy comparison.

## Docker Setup
- Base image: `python:3.11-slim`
- For GPU: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` (commented in Dockerfile)
- docker-compose uses `deploy.resources.reservations.devices` for GPU passthrough
- Requires: NVIDIA Container Toolkit on host (`nvidia-container-runtime`)

## Local Dev Commands
```bash
# Activate venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dev server (auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test pipeline with real photos
python tests/demo.py

# Test STT with microphone
python tests/mic_test.py --seconds 5

# Run via Docker Compose (GPU config in docker-compose.yml)
docker compose up --build
```

## Configuration (.env)
```env
MOCK_STT=false            # faster-whisper tiny (CUDA)
MOCK_SAM=false            # FastSAM-s (ultralytics)
MOCK_BLIP=false           # blip-image-captioning-large
MOCK_OLLAMA=false         # qwen2.5:7b via Ollama
MOCK_TTS=false            # Edge TTS EmelNeural +30%

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT_SECONDS=30

BLIP_MODEL=Salesforce/blip-image-captioning-large
TTS_VOICE=tr-TR-EmelNeural
TTS_RATE=+30%
```
