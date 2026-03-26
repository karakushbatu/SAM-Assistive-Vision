"""
SAM Assistive Vision -- FastAPI Entry Point
===========================================
WebSocket flow (one frame cycle):
    1. Android  --[binary: JPEG bytes]-->  /ws/vision
    2. Server   runs 5-stage pipeline
    3. Server   --[text:   JSON metadata]-->  Android
    4. Server   --[binary: MP3 bytes]-->      Android  (streamed in chunks if TTS_STREAMING=true)

Session state (per connection):
    - current_mode: "scene" | "ocr"   (persists across frames until changed)
    - conversation_history: last 3 exchanges (for context-aware follow-ups)
    - last_result: cached last PipelineResult for replay
    - pipeline_lock: per-connection asyncio.Lock (skips frame if pipeline busy)
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.logger import logger
from services.ai_pipeline import run_full_pipeline
from services.intent_service import detect_intent, should_offer_ocr


# ---------------------------------------------------------------------------
# Connection counter (thread-safe via asyncio — single event loop)
# ---------------------------------------------------------------------------

_active_connections: int = 0


# ---------------------------------------------------------------------------
# Startup: warm up all models
# ---------------------------------------------------------------------------

async def _warmup_stt() -> None:
    if settings.mock_stt:
        logger.info("STT warmup skipped (mock mode)")
        return
    from services.stt_service import load_stt_model
    await asyncio.to_thread(load_stt_model)


async def _warmup_sam() -> None:
    if settings.mock_sam:
        logger.info("SAM warmup skipped (mock mode)")
        return
    from services.sam_service import load_sam_model
    await asyncio.to_thread(load_sam_model)


async def _warmup_blip() -> None:
    if settings.mock_blip:
        logger.info("BLIP warmup skipped (mock mode)")
        return
    from services.blip_service import load_blip_model
    await asyncio.to_thread(load_blip_model)


async def _warmup_ollama() -> None:
    if settings.mock_ollama:
        logger.info("Ollama warmup skipped (mock mode)")
        return
    import httpx
    logger.info("Warming up Ollama model '%s' ...", settings.ollama_model)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": settings.ollama_model, "prompt": "Merhaba",
                      "stream": False, "options": {"num_predict": 1}},
            )
        logger.info("Ollama warmup complete.")
    except Exception as e:
        logger.warning("Ollama warmup failed (will retry on first request): %s", e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Enable Ollama Flash Attention — must also be set in host env before Ollama starts
    os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
    await _warmup_stt()
    await _warmup_sam()
    await _warmup_blip()
    await _warmup_ollama()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

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
    """Liveness + readiness probe with Ollama connectivity, GPU stats, and model status."""
    ollama_status = await _probe_ollama()

    # GPU VRAM stats (lazy import — avoids error if torch not installed)
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "vram_used_mb":  round(torch.cuda.memory_allocated() / 1024 ** 2),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2),
                "device":        torch.cuda.get_device_name(0),
            }
        else:
            gpu_info = {"available": False}
    except ImportError:
        gpu_info = {"available": "torch not installed"}

    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "active_connections": _active_connections,
        "mocks": {
            "stt":    settings.mock_stt,
            "sam":    settings.mock_sam,
            "blip":   settings.mock_blip,
            "ollama": settings.mock_ollama,
            "tts":    settings.mock_tts,
        },
        "models_loaded": _get_models_loaded(),
        "tts_voice":     settings.tts_voice,
        "tts_rate":      settings.tts_rate,
        "tts_streaming": settings.tts_streaming,
        "max_frame_size": settings.max_frame_size,
        "ollama":        ollama_status,
        "gpu":           gpu_info,
    }


@app.get("/", response_class=JSONResponse)
async def root():
    return {"message": f"Welcome to {settings.app_name} v{settings.app_version}"}


async def _probe_ollama() -> dict:
    if settings.mock_ollama:
        return {"status": "mock"}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            t = time.monotonic()
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            latency_ms = round((time.monotonic() - t) * 1000)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return {"status": "ok", "latency_ms": latency_ms,
                    "configured_model": settings.ollama_model,
                    "available_models": models,
                    "model_ready": settings.ollama_model in models}
    except Exception as e:
        return {"status": "unreachable", "error": str(e),
                "configured_model": settings.ollama_model}


def _get_models_loaded() -> dict:
    """Check which model singletons have been loaded into memory."""
    result = {}
    if not settings.mock_stt:
        try:
            from services.stt_service import _model as _stt_m
            result["stt"] = _stt_m is not None
        except Exception:
            result["stt"] = False
    if not settings.mock_sam:
        try:
            from services.sam_service import _sam_model
            result["sam"] = _sam_model is not None
        except Exception:
            result["sam"] = False
    if not settings.mock_blip:
        try:
            from services.blip_service import _model as _blip_m
            result["blip"] = _blip_m is not None
        except Exception:
            result["blip"] = False
    return result


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

_SILENT_MP3 = bytes([
    0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])  # Minimal valid MP3 frame for mute / busy responses


async def _ping_loop(websocket: WebSocket) -> None:
    """Send a keepalive ping every 20s to prevent WebSocket timeout."""
    try:
        while True:
            await asyncio.sleep(20)
            await websocket.send_text(json.dumps({"status": "ping"}))
    except Exception:
        pass  # Connection closed — stop silently


@app.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint.

    Per-frame protocol:
        RECEIVE  binary  -- raw JPEG/PNG image bytes from Android camera
        SEND     text    -- JSON: {status, frame_id, description, user_query,
                                   mode, ocr_available, total_latency_ms, pipeline}
        SEND     binary  -- MP3 audio bytes (streamed in chunks if TTS_STREAMING=true)

    Busy frame response (pipeline still processing previous frame):
        SEND     text    -- {"status": "busy", "frame_id": ..., "frame_skipped": true}
        SEND     binary  -- silent MP3 frame

    Session state (resets on disconnect):
        current_mode         -- "scene" or "ocr"
        conversation_history -- last 3 [user_query, description] pairs
        last_result          -- cached last PipelineResult for replay
        pipeline_lock        -- ensures one pipeline run at a time per connection
    """
    global _active_connections

    await websocket.accept()
    _active_connections += 1
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("WebSocket connected | client=%s | total=%d", client_host, _active_connections)

    # Per-session state
    current_mode = "scene"
    conversation_history: list[dict] = []
    last_result = None
    pipeline_lock = asyncio.Lock()

    ping_task = asyncio.create_task(_ping_loop(websocket))

    try:
        while True:
            image_bytes: bytes = await websocket.receive_bytes()
            logger.debug("Frame received | size=%d bytes | mode=%s", len(image_bytes), current_mode)

            # --- Frame skip: drop frame if pipeline is still processing previous one ---
            if pipeline_lock.locked():
                skip_id = str(uuid.uuid4())
                logger.debug("Pipeline busy — skipping frame | frame_id=%s", skip_id)
                await websocket.send_text(json.dumps({
                    "status": "busy",
                    "frame_id": skip_id,
                    "frame_skipped": True,
                }))
                await websocket.send_bytes(_SILENT_MP3)
                continue

            async with pipeline_lock:
                try:
                    use_streaming = settings.tts_streaming and not settings.mock_tts

                    result = await run_full_pipeline(
                        image_bytes,
                        audio_bytes=None,
                        mode=current_mode,
                        history=conversation_history,
                        skip_tts=use_streaming,
                    )

                    # Detect intent from STT text this frame returned
                    intent_info = detect_intent(result.metadata.get("user_query"))
                    intent = intent_info["intent"]

                    # Update mode based on intent
                    if intent == "ocr":
                        current_mode = "ocr"
                    elif intent in ("scene", "camera"):
                        current_mode = "scene"

                    # Handle replay — resend last response without reprocessing
                    if intent == "replay" and last_result is not None:
                        await websocket.send_text(
                            json.dumps(last_result.metadata, ensure_ascii=False)
                        )
                        await websocket.send_bytes(last_result.audio_bytes)
                        continue

                    # Handle mute — send JSON but silent audio
                    if intent == "mute":
                        result.metadata["muted"] = True
                        await websocket.send_text(
                            json.dumps(result.metadata, ensure_ascii=False)
                        )
                        await websocket.send_bytes(_SILENT_MP3)
                        continue

                    # Proactive OCR offer flag
                    blip_caption = result.metadata.get("pipeline", {}).get("blip", {}).get("caption", "")
                    ocr_available = current_mode == "scene" and should_offer_ocr(blip_caption)
                    result.metadata["ocr_available"] = ocr_available
                    result.metadata["mode"] = current_mode

                    # Update conversation history (keep last 3)
                    if result.metadata.get("description"):
                        conversation_history.append({
                            "user": result.metadata.get("user_query"),
                            "assistant": result.metadata["description"],
                        })
                        if len(conversation_history) > 3:
                            conversation_history = conversation_history[-3:]

                    # Message 1: JSON metadata (sent immediately in both modes)
                    await websocket.send_text(
                        json.dumps(result.metadata, ensure_ascii=False)
                    )

                    # Message 2: MP3 audio
                    if use_streaming:
                        # Stream chunks as they are generated — client hears audio sooner
                        from services.tts_service import stream_tts
                        audio_chunks: list[bytes] = []
                        async for chunk in stream_tts(result.metadata["description"]):
                            audio_chunks.append(chunk)
                            await websocket.send_bytes(chunk)
                        result.audio_bytes = b"".join(audio_chunks)
                    else:
                        await websocket.send_bytes(result.audio_bytes)

                    last_result = result

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
        ping_task.cancel()
        _active_connections -= 1
        logger.info("WebSocket connection closed | client=%s | total=%d",
                    client_host, _active_connections)
