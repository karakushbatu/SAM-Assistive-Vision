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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.logger import logger
from services.ai_pipeline import run_full_pipeline

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time visual assistant backend for the visually impaired.",
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Liveness probe -- used by Docker health checks and monitoring."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "mock_pipeline": settings.mock_pipeline,
        "tts_voice": settings.tts_voice,
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
