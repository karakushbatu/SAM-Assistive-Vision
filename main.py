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
import struct
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from core.config import settings
from core.logger import logger
from services.ai_pipeline import run_full_pipeline
from services.intent_service import detect_intent, should_offer_ocr
from services.stt_service import run_stt


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


async def _warmup_ocr() -> None:
    if settings.mock_ollama or settings.ocr_backend != "paddleocr":
        logger.info("OCR warmup skipped | mock=%s | backend=%s", settings.mock_ollama, settings.ocr_backend)
        return
    from services.ocr_service import load_ocr_engine
    await asyncio.to_thread(load_ocr_engine)


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
                      "stream": False, "keep_alive": settings.ollama_keep_alive,
                      "options": {"num_predict": 1}},
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
    await _warmup_ocr()
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

    models_loaded = _get_models_loaded()
    ready = (
        (settings.mock_ollama or ollama_status.get("model_ready") is True)
        and (settings.mock_ollama or ollama_status.get("scene_model_ready") is True)
        and (settings.mock_ollama or ollama_status.get("ocr_summary_model_ready") is True)
        and (settings.mock_ollama or ollama_status.get("ocr_model_ready") is True)
        and all(models_loaded.values())
    )

    return {
        "status": "healthy",
        "ready": ready,
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
        "models_loaded": models_loaded,
        "tts_voice":     settings.tts_voice,
        "tts_rate":      settings.tts_rate,
        "tts_streaming": settings.tts_streaming,
        "max_frame_size": settings.max_frame_size,
        "ollama":        ollama_status,
        "ocr_backend":   settings.ocr_backend,
        "gpu":           gpu_info,
    }


@app.get("/", response_class=JSONResponse)
async def root():
    return {"message": f"Welcome to {settings.app_name} v{settings.app_version}"}


@app.get("/contract", response_class=JSONResponse)
async def contract():
    """Machine-readable integration contract for Android handoff."""
    limits = {
        "max_ws_frame_bytes": settings.max_ws_frame_bytes,
        "max_frame_size": settings.max_frame_size,
        "stt_min_audio_bytes": settings.stt_min_audio_bytes,
        "stt_max_audio_seconds": settings.stt_max_audio_seconds,
    }
    endpoints = {
        "health": "GET /health",
        "contract": "GET /contract",
        "vision_websocket": "WS /ws/vision",
    }
    protocol = {
        "legacy_image": "raw JPEG/PNG bytes",
        "audio_image_envelope": (
            "[4-byte little-endian audio_length]"
            "[WAV audio bytes][JPEG/PNG image bytes]"
        ),
        "response_order": (
            "text JSON status event, then MP3 binary response; streaming TTS sends "
            "MP3 chunks followed by text JSON status=audio_end"
        ),
    }
    return {
        "version": 1,
        "endpoints": endpoints,
        "protocol": protocol,
        "limits": limits,
        "websocket": {
            "path": "/ws/vision",
            "input_binary_formats": {
                "legacy_image": protocol["legacy_image"],
                "audio_image_envelope": protocol["audio_image_envelope"],
            },
            "limits": limits,
            "server_text_events": [
                "ok", "busy", "error", "ping", "audio_end",
            ],
            "successful_flow": [
                "client sends binary frame",
                "server sends text JSON with status=ok",
                "server sends one MP3 binary if audio_streaming=false",
                "server sends MP3 binary chunks followed by status=audio_end if audio_streaming=true",
            ],
        },
        "metadata_fields": {
            "status": "ok | busy | error | ping | audio_end",
            "frame_id": "uuid string on frame-bound events",
            "mode": "scene | ocr",
            "description": "Turkish text for TTS/display",
            "user_query": "Turkish STT text or null",
            "ocr_available": "boolean OCR offer flag",
            "audio_streaming": settings.tts_streaming and not settings.mock_tts,
            "pipeline": "present for full pipeline ok responses only",
        },
    }


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
                    "scene_model": settings.scene_ollama_model,
                    "ocr_summary_model": settings.ocr_summary_model,
                    "ocr_backend": settings.ocr_backend,
                    "configured_ocr_model": settings.ocr_model,
                    "available_models": models,
                    "model_ready": settings.ollama_model in models,
                    "scene_model_ready": settings.scene_ollama_model in models,
                    "ocr_summary_model_ready": settings.ocr_summary_model in models,
                    "ocr_model_ready": (
                        True if settings.ocr_backend == "paddleocr"
                        else settings.ocr_model in models
                    )}
    except Exception as e:
        return {"status": "unreachable", "error": str(e),
                "configured_model": settings.ollama_model,
                "scene_model": settings.scene_ollama_model,
                "ocr_summary_model": settings.ocr_summary_model,
                "ocr_backend": settings.ocr_backend,
                "configured_ocr_model": settings.ocr_model}


def _get_models_loaded() -> dict:
    """Check which model singletons have been loaded into memory."""
    result = {}
    if not settings.mock_stt:
        try:
            from services.stt_service import _stt_model
            result["stt"] = _stt_model is not None
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
    if not settings.mock_ollama and settings.ocr_backend == "paddleocr":
        try:
            from services.ocr_service import _ocr_engine
            result["ocr"] = _ocr_engine is not None
        except Exception:
            result["ocr"] = False
    return result


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

_SILENT_MP3 = bytes([
    0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])  # Minimal valid MP3 frame for mute / busy responses

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG"


def _is_image_bytes(payload: bytes) -> bool:
    return payload[:3] == _JPEG_MAGIC or payload[:4] == _PNG_MAGIC


def _parse_frame_payload(payload: bytes) -> tuple[bytes, bytes | None, str]:
    """
    Parse WebSocket binary payload.

    Supported formats:
        1. Legacy: raw JPEG/PNG bytes.
        2. Envelope: [4-byte little-endian audio_len][WAV audio bytes][JPEG/PNG bytes].
    """
    if len(payload) > settings.max_ws_frame_bytes:
        raise ValueError(
            f"Frame too large: {len(payload)} bytes exceeds "
            f"MAX_WS_FRAME_BYTES={settings.max_ws_frame_bytes}"
        )

    if _is_image_bytes(payload):
        return payload, None, "legacy_image"

    if len(payload) < 4:
        raise ValueError("Invalid frame payload: expected image bytes or audio envelope")

    audio_len = struct.unpack("<I", payload[:4])[0]
    if audio_len > len(payload) - 4:
        raise ValueError(
            f"Invalid audio envelope: audio_len={audio_len}, payload_size={len(payload)}"
        )

    audio_bytes = payload[4:4 + audio_len] if audio_len else None
    image_bytes = payload[4 + audio_len:]

    if not _is_image_bytes(image_bytes):
        raise ValueError("Invalid frame payload: envelope does not contain JPEG/PNG image bytes")

    if audio_bytes is not None and len(audio_bytes) < settings.stt_min_audio_bytes:
        logger.debug("Ignoring short audio payload | bytes=%d", len(audio_bytes))
        audio_bytes = None

    return image_bytes, audio_bytes, "audio_image_envelope"


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
            payload: bytes = await websocket.receive_bytes()
            try:
                image_bytes, audio_bytes, protocol = _parse_frame_payload(payload)
            except ValueError as payload_error:
                logger.warning("Invalid frame payload: %s", payload_error)
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": str(payload_error),
                }))
                await websocket.send_bytes(_SILENT_MP3)
                continue
            logger.debug(
                "Frame received | protocol=%s | payload=%d bytes | image=%d bytes | audio=%s | mode=%s",
                protocol,
                len(payload),
                len(image_bytes),
                len(audio_bytes) if audio_bytes is not None else None,
                current_mode,
            )

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
                    stt_result = await run_stt(audio_bytes) if audio_bytes is not None else None

                    # Fast-path voice-only commands before expensive vision stages.
                    intent_info = detect_intent(
                        stt_result["text"] if stt_result is not None else None
                    )
                    intent = intent_info["intent"]
                    logger.info(
                        "Intent detected | text=%r | intent=%s | mode_before=%s",
                        stt_result["text"] if stt_result is not None else None,
                        intent,
                        current_mode,
                    )

                    if intent == "replay" and last_result is not None:
                        await websocket.send_text(
                            json.dumps(last_result.metadata, ensure_ascii=False)
                        )
                        await websocket.send_bytes(last_result.audio_bytes or _SILENT_MP3)
                        continue

                    if intent == "mute":
                        mute_id = str(uuid.uuid4())
                        await websocket.send_text(json.dumps({
                            "status": "ok",
                            "frame_id": mute_id,
                            "mode": current_mode,
                            "muted": True,
                            "description": "",
                            "user_query": stt_result["text"] if stt_result else None,
                            "ocr_available": False,
                            "audio_streaming": False,
                            "total_latency_ms": stt_result["latency_ms"] if stt_result else 0.0,
                        }, ensure_ascii=False))
                        await websocket.send_bytes(_SILENT_MP3)
                        continue

                    if intent == "camera":
                        current_mode = "scene"
                        camera_id = str(uuid.uuid4())
                        await websocket.send_text(json.dumps({
                            "status": "ok",
                            "frame_id": camera_id,
                            "mode": current_mode,
                            "camera": "ready",
                            "description": "Kamera hazır.",
                            "user_query": stt_result["text"] if stt_result else None,
                            "ocr_available": False,
                            "audio_streaming": False,
                            "total_latency_ms": stt_result["latency_ms"] if stt_result else 0.0,
                        }, ensure_ascii=False))
                        await websocket.send_bytes(_SILENT_MP3)
                        continue

                    if intent == "ocr":
                        current_mode = "ocr"
                    elif intent == "scene":
                        current_mode = "scene"

                    result = await run_full_pipeline(
                        image_bytes,
                        audio_bytes=audio_bytes,
                        mode=current_mode,
                        history=conversation_history,
                        skip_tts=use_streaming,
                        precomputed_stt=stt_result,
                    )

                    # Detect intent from STT text this frame returned
                    intent_info = detect_intent(result.metadata.get("user_query"))
                    intent = intent_info["intent"]
                    logger.info(
                        "Intent re-check | text=%r | intent=%s | mode_before=%s",
                        result.metadata.get("user_query"),
                        intent,
                        current_mode,
                    )

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
                    result.metadata["audio_streaming"] = use_streaming
                    if current_mode == "ocr":
                        result.metadata["ocr_status"] = result.metadata["pipeline"]["ocr"]["status"]

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
                        # Stream chunks as they are generated - client hears audio sooner
                        from services.tts_service import get_silent_audio_bytes, stream_tts
                        audio_chunks: list[bytes] = []
                        try:
                            async for chunk in stream_tts(result.metadata["description"]):
                                audio_chunks.append(chunk)
                                await websocket.send_bytes(chunk)
                        except Exception as tts_error:
                            logger.warning("Streaming TTS failed, falling back to silent audio: %s", tts_error)
                            fallback_chunk = get_silent_audio_bytes()
                            audio_chunks = [fallback_chunk]
                            result.metadata["pipeline"]["tts"]["fallback"] = "silent_mock"
                            result.metadata["pipeline"]["tts"]["error"] = str(tts_error)
                            await websocket.send_bytes(fallback_chunk)

                        result.audio_bytes = b"".join(audio_chunks)
                        audio_size = len(result.audio_bytes)
                        result.metadata["pipeline"]["tts"]["audio_size_bytes"] = audio_size
                        await websocket.send_text(json.dumps({
                            "status": "audio_end",
                            "frame_id": result.metadata["frame_id"],
                            "audio_size_bytes": audio_size,
                        }))
                    else:
                        await websocket.send_bytes(result.audio_bytes)

                    last_result = result

                except Exception as pipeline_error:
                    logger.error("Pipeline error: %s", pipeline_error, exc_info=True)
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "message": f"Pipeline failed: {str(pipeline_error)}",
                    }))
                    await websocket.send_bytes(_SILENT_MP3)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected | client=%s", client_host)
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e, exc_info=True)
    finally:
        ping_task.cancel()
        _active_connections -= 1
        logger.info("WebSocket connection closed | client=%s | total=%d",
                    client_host, _active_connections)
