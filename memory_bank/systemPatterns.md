# System Patterns

## WebSocket Communication Contract

### Client → Server
- Binary message: raw JPEG/PNG image bytes (1 frame per second from camera)
- Android waits for BOTH responses before sending next frame (backpressure)

### Server → Client (JSON)
```json
{
  "status": "ok",
  "frame_id": "uuid-string",
  "pipeline": {
    "sam": { "masks_found": 3, "latency_ms": 120 },
    "classifier": { "labels": ["person", "chair", "door"], "latency_ms": 45 },
    "ollama": { "description": "A person is standing near a door.", "latency_ms": 800 }
  },
  "total_latency_ms": 965,
  "description": "A person is standing near a door."
}
```

### Error Response (JSON)
```json
{
  "status": "error",
  "message": "Pipeline failed at SAM stage."
}
```

## Pipeline Architecture Pattern
- Each pipeline stage is an `async def` function
- Stages are awaited **sequentially** (output of one feeds into next)
- Each stage returns a typed Python dict
- The WebSocket handler in main.py orchestrates the pipeline and serializes results

## Directory Layout Convention
```
/app
  main.py               ← FastAPI app + WebSocket router
  /api
    websocket.py        ← WebSocket endpoint handler (future refactor)
  /services
    ai_pipeline.py      ← All pipeline stage functions (mock now, real later)
    sam_service.py      ← (Phase 2) FastSAM/MobileSAM integration
    classifier_service.py ← (Phase 3) CLIP/ResNet integration
    ollama_service.py   ← (Phase 4) Ollama HTTP client
  /core
    config.py           ← Env-based configuration (Settings class)
    logger.py           ← Centralized logging setup
  /docker
    Dockerfile
    docker-compose.yml
  requirements.txt
```

## Configuration Pattern
- Use `pydantic-settings` with a `Settings` class
- All config comes from environment variables or `.env` file
- Never hard-code ports, model names, or URLs
