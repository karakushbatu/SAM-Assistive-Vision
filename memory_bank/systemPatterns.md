# System Patterns

## WebSocket Communication Contract

### Client → Server
- Binary message: raw JPEG/PNG image bytes (1 frame per second from camera)
- Android waits for BOTH responses before sending next frame (backpressure)

### Server → Client (JSON text message)
```json
{
  "status": "ok",
  "frame_id": "uuid-string",
  "mode": "scene",
  "description": "Elinizde Kreon sindirim ilacı kutusu var.",
  "user_query": null,
  "ocr_available": true,
  "total_latency_ms": 3200.5,
  "pipeline": {
    "stt":    { "text": null, "latency_ms": 35.2 },
    "sam":    { "masks_found": 4, "latency_ms": 23.1 },
    "blip":   { "caption": "kreon pills box; white box in background", "latency_ms": 1450.3 },
    "ollama": { "latency_ms": 2100.8 },
    "tts":    { "audio_size_bytes": 18432, "latency_ms": 290.1 }
  }
}
```

### Server → Client (binary message)
- MP3 audio bytes (Edge TTS output, immediately after JSON)

### Error Response (JSON)
```json
{
  "status": "error",
  "message": "Pipeline failed: Ollama is unreachable."
}
```

## Pipeline Architecture Pattern
- Each pipeline stage is an `async def` function in its service file
- Stages are awaited **sequentially** in `ai_pipeline.py` (output feeds into next)
- Each stage returns a typed Python dict
- Mock/real switching via `.env` flags — no code changes needed
- Model loading at startup via `lifespan()` in main.py

## Per-Stage Data Flow

```
image_bytes (JPEG)
    │
    ▼
[Stage 1: STT]  audio_bytes? → {"text": str|None, "latency_ms": float}
    │
    ▼
[Stage 3: SAM]  image_bytes → {"masks_found": int, "bounding_boxes": list,
    │                           "crops": list[bytes], "latency_ms": float}
    ▼
[Stage 4: BLIP] image_bytes + sam_result → {"caption": str, "latency_ms": float}
    │             (captions each crop separately, joins top 3 with "; ")
    ▼
[Stage 5: Ollama] blip_result + user_query? + history → {"description": str,
    │                                                      "ocr_recommended": bool,
    │                                                      "latency_ms": float}
    │              (2 calls: OCR yes/no → description → post-process OCR offer)
    ▼
[Stage 6: TTS]  ollama_result → {"audio_bytes": bytes, "audio_size_bytes": int,
                                  "latency_ms": float}
```

## Ollama 2-Call Pattern

```python
# Call 1: Fast OCR check (temperature=0.0, num_predict=3)
ocr_raw = await ollama.post(system=_SYSTEM_OCR_CHECK, prompt=caption)
ocr_recommended = ocr_raw.startswith("yes")

# Call 2: Turkish description (temperature=0.2, num_predict=100)
description = await ollama.post(system=_SYSTEM_PROMPT, prompt=user_prompt)

# Post-processing — not LLM-generated, keeps description fast
if ocr_recommended:
    description = description.rstrip(".!?") + "." + _OCR_OFFER
```

## Intent Service Pattern
- `detect_intent(stt_text)` → `{intent, keeps_context, raw_text}`
- Turkish keyword matching (no ML model) — fast and reliable
- Intents: "scene" | "ocr" | "replay" | "mute" | "camera" | None
- None = free-form question → pass as user_query to current pipeline mode

## Directory Layout
```
/SAM-Assistive-Vision
  main.py                    ← FastAPI app + WebSocket endpoint + lifespan warmups
  /services
    ai_pipeline.py           ← Pipeline orchestrator (Stage 1→3→4→5→6)
    stt_service.py           ← Stage 1: faster-whisper tiny STT
    sam_service.py           ← Stage 3: FastSAM-s segmentation + crop extraction
    blip_service.py          ← Stage 4: BLIP multi-crop image captioning
    ollama_service.py        ← Stage 5: Ollama LLM (2-call: OCR + description)
    tts_service.py           ← Stage 6: Edge TTS audio generation
    intent_service.py        ← Turkish keyword → pipeline intent dispatcher
    classifier_service.py    ← SUPERSEDED by blip_service.py (kept for reference)
  /core
    config.py                ← Pydantic-settings (all config from .env)
    logger.py                ← Centralized logging
  /docker
    Dockerfile
    docker-compose.yml       ← API + Ollama, NVIDIA GPU passthrough configured
  /tests
    demo.py                  ← Real photo pipeline test
    mic_test.py              ← Microphone → STT demo
    test_websocket.py        ← WebSocket smoke test
    /sample_images           ← Test photos (ilaç, yulaf, FPS game, etc.)
  /memory_bank               ← Project Memory Bank (this folder)
  requirements.txt
  .env
  FastSAM-s.pt               ← FastSAM model weights (local)
```

## Configuration Pattern
- Use `pydantic-settings` with a `Settings` class
- All config comes from environment variables or `.env` file
- Never hard-code ports, model names, or URLs
- Mock flags per service: `MOCK_STT`, `MOCK_SAM`, `MOCK_BLIP`, `MOCK_OLLAMA`, `MOCK_TTS`
- Model swap: change env var only — service code has no model name hard-coded
