# Tech Context

## Python Environment
- Active local environment: `.venv`
- Python version: `3.12.10`

## Key Dependencies

| Package | Purpose |
|---|---|
| fastapi | HTTP + WebSocket framework |
| uvicorn[standard] | ASGI server |
| pydantic-settings | Env-based config |
| httpx | Async HTTP client for Ollama |
| torch 2.11.0+cu128 | CUDA tensor runtime |
| transformers 5.x | BLIP caption model |
| ultralytics | FastSAM-s segmentation |
| faster-whisper | STT |
| edge-tts | Turkish TTS |
| sounddevice | Local microphone capture for demo/testing |

## Hardware
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- VRAM: `16 GB`
- Verified local state on 2026-04-16:
  - `torch.cuda.is_available() == True`

## Cached / Installed Models
- Hugging Face:
  - `Salesforce/blip-image-captioning-large`
  - `Systran/faster-whisper-tiny`
  - `Systran/faster-whisper-small`
- Local file:
  - `FastSAM-s.pt`
- Ollama:
  - `qwen2.5:7b`
  - `qwen3:4b`
  - `glm-ocr:latest`
  - `gemma3:4b`
  - `llama3.2:3b`

## Active Runtime Decisions
- Vision stack:
  - `FastSAM-s`
  - `BLIP-large`
- LLM stack:
  - default: `qwen2.5:7b`
- OCR stack:
  - active default: `glm-ocr:latest`
  - optional experimental path: `PaddleOCR`
- STT stack:
  - default: `tiny`
- TTS stack:
  - current local fallback: `tr-TR-AhmetNeural`
  - current local rate profile: `+10%`
  - future product direction: stronger premium TTS provider for more natural voice quality

## Local LLM Benchmark Snapshot - 2026-04-16
- `qwen2.5:7b`
  - scene prompt latency: about `5265 ms`
  - OCR summary latency: about `656 ms`
  - best usable quality in current flow
- `qwen3:4b`
  - scene prompt latency: about `3344 ms`
  - OCR summary latency: about `813 ms`
  - current output was empty or not usable in this Ollama flow

Decision:
- Keep `qwen2.5:7b` as default.

## OCR Validation Snapshot - 2026-04-23
- PaddleOCR was integrated and evaluated.
- Current Windows runtime behavior was not good enough for active use:
  - CPU fallback
  - weak or empty OCR extraction on local sample set
  - worse latency than the previous path

Decision:
- Keep PaddleOCR only as an optional experiment.
- Active local OCR/demo path remains:
  - `OCR_BACKEND=ollama_vision`
  - `OCR_MODEL=glm-ocr:latest`

## Optimized Local `.env` Profile
```env
MODEL_DEVICE=cuda
HF_LOCAL_FILES_ONLY=true

MOCK_STT=false
MOCK_SAM=false
MOCK_BLIP=false
MOCK_OLLAMA=false
MOCK_TTS=false

STT_MODEL=tiny
SAM_MAX_CROPS=2
BLIP_MAX_CAPTIONS=2
BLIP_MAX_NEW_TOKENS=24

SCENE_OLLAMA_MODEL=qwen2.5:7b
OCR_SUMMARY_MODEL=qwen2.5:7b
OLLAMA_NUM_PREDICT=64
OLLAMA_TEMPERATURE=0.1
OLLAMA_KEEP_ALIVE=30m

OCR_BACKEND=ollama_vision
OCR_MODEL=glm-ocr:latest
```
