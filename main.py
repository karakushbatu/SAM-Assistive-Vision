"""
SAM Assistive Vision -- FastAPI Entry Point
===========================================
WebSocket flow (one frame cycle):
    1. Android  --[binary: JPEG bytes]-->  /ws/vision
    2. Server   runs 4-stage pipeline
    3. Server   --[text:   JSON metadata]-->  Android   (description, latencies)
    4. Server   --[binary: MP3 bytes]-->      Android   (audio to play)
"""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.logger import logger
from services.ai_pipeline import run_full_pipeline


# ---------------------------------------------------------------------------
# B1: Ollama Warmup — pre-load model into RAM on server startup
# ---------------------------------------------------------------------------

async def _warmup_ollama() -> None:
    """Send a minimal dummy request so the first real frame isn't slow."""
    if settings.mock_ollama:
        logger.info("Ollama warmup skipped (mock mode)")
        return

    import httpx

    logger.info("Warming up Ollama model '%s' ...", settings.ollama_model)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": "Merhaba",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
        logger.info("Ollama warmup complete — model is ready.")
    except Exception as e:
        logger.warning(
            "Ollama warmup failed (will retry on first request): %s", e
        )


# ---------------------------------------------------------------------------
# B4: Ollama health probe — used by /health endpoint
# ---------------------------------------------------------------------------

async def _probe_ollama() -> dict:
    """Check Ollama reachability and list loaded models. Max 3s wait."""
    if settings.mock_ollama:
        return {"status": "mock", "models": []}

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            t = time.monotonic()
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            latency_ms = round((time.monotonic() - t) * 1000)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return {
                "status": "ok",
                "latency_ms": latency_ms,
                "configured_model": settings.ollama_model,
                "available_models": models,
                "model_ready": settings.ollama_model in models,
            }
    except Exception as e:
        return {
            "status": "unreachable",
            "error": str(e),
            "configured_model": settings.ollama_model,
        }


# ---------------------------------------------------------------------------
# App initialization with lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _warmup_ollama()
    yield  # server runs here
    # (shutdown logic can go here if needed in the future)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time visual assistant backend for the visually impaired.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Detailed liveness + readiness probe.
    Returns Ollama connectivity status, loaded models, and mock flags.
    """
    ollama_status = await _probe_ollama()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "mocks": {
            "sam":        settings.mock_sam,
            "classifier": settings.mock_classifier,
            "ollama":     settings.mock_ollama,
            "tts":        settings.mock_tts,
        },
        "tts_voice": settings.tts_voice,
        "tts_rate":  settings.tts_rate,
        "ollama":    ollama_status,
    }


@app.get("/", response_class=JSONResponse)
async def root():
    return {"message": f"Welcome to {settings.app_name} v{settings.app_version}"}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint.

    Per-frame protocol:
        RECEIVE  binary  -- raw JPEG/PNG image bytes from Android camera
        SEND     text    -- JSON string: {status, frame_id, description,
                                          total_latency_ms, pipeline:{...}}
        SEND     binary  -- MP3 audio bytes (Android plays immediately)

    The connection stays open across frames -- no reconnect needed.
    Android should wait for both messages before sending the next frame.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("WebSocket connected | client=%s", client_host)

    try:
        while True:
            image_bytes: bytes = await websocket.receive_bytes()
            logger.debug("Frame received | size=%d bytes", len(image_bytes))

            try:
                result = await run_full_pipeline(image_bytes)

                # Message 1: JSON metadata (text)
                await websocket.send_text(
                    json.dumps(result.metadata, ensure_ascii=False)
                )

                # Message 2: MP3 audio (binary)
                await websocket.send_bytes(result.audio_bytes)

            except Exception as pipeline_error:
                logger.error("Pipeline error: %s", pipeline_error, exc_info=True)
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": f"Pipeline failed: {str(pipeline_error)}",
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected | client=%s", client_host)
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e, exc_info=True)
    finally:
        logger.info("WebSocket connection closed | client=%s", client_host)
