"""
AI Pipeline -- Orchestrator
============================
This file ONLY coordinates the pipeline stages. It contains NO model logic.

Stage map (matches pipeline.jpg):
    Stage 1: STT    (stt_service)   -- voice input -> transcribed text (optional)
    Stage 2: Camera input           -- JPEG bytes received by WebSocket in main.py
    Stage 3: SAM    (sam_service)   -- image -> segmentation masks
    Stage 4: BLIP   (blip_service)  -- image + masks -> scene caption (text)
    Stage 5: LLM    (ollama_service)-- caption [+ query + history] -> Turkish description
    Stage 6: TTS    (tts_service)   -- text -> MP3 audio

To swap any model/engine: edit the corresponding service file.
To change pipeline order or add a stage: edit only this file.
"""

import time
import uuid
from typing import Any

from core.logger import logger
from services.stt_service import run_stt
from services.sam_service import run_sam
from services.blip_service import run_blip
from services.ollama_service import run_ollama
from services.tts_service import run_tts


class PipelineResult:
    """
    Container for the two WebSocket messages sent back per frame:
        .metadata    -- dict  -> send as JSON text message
        .audio_bytes -- bytes -> send as binary message immediately after
    """
    def __init__(self, metadata: dict[str, Any], audio_bytes: bytes) -> None:
        self.metadata = metadata
        self.audio_bytes = audio_bytes


async def run_full_pipeline(
    image_bytes: bytes,
    audio_bytes: bytes | None = None,
    mode: str = "scene",
    history: list[dict] | None = None,
    skip_tts: bool = False,
) -> PipelineResult:
    """
    Execute the 5-stage pipeline sequentially on one image frame.

    Args:
        image_bytes: Raw JPEG/PNG bytes from the Android WebSocket client.
        audio_bytes: Optional WAV audio bytes (user voice query).
                     Pass None when no voice input for this frame.
        mode:        "scene" (default) or "ocr". Passed through to metadata.
        history:     Conversation history [{user, assistant}, ...] for context.
                     Keep last 3 exchanges. Pass None or [] for no context.
        skip_tts:    If True, skip Stage 6 (TTS). Used by streaming mode in
                     main.py so the WebSocket handler can stream audio directly.

    Returns:
        PipelineResult:
            .metadata    -> JSON-serializable dict with description + latencies
            .audio_bytes -> MP3 bytes (empty b"" when skip_tts=True)
    """
    frame_id = str(uuid.uuid4())
    t_start = time.monotonic()
    logger.info("Pipeline START | frame_id=%s | mode=%s", frame_id, mode)

    # Stage 1: STT — transcribe voice query (skipped instantly if audio=None)
    stt_result = await run_stt(audio_bytes)

    # Stage 3: SAM — segment the image into regions
    sam_result = await run_sam(image_bytes)

    # Stage 4: BLIP — caption the scene (image -> text)
    blip_result = await run_blip(image_bytes, sam_result)

    # Stage 5: LLM — generate Turkish description from caption + query + history
    ollama_result = await run_ollama(
        blip_result,
        user_query=stt_result["text"],
        history=history or [],
    )

    # Stage 6: TTS — skip if caller wants to stream audio directly
    if skip_tts:
        tts_result = {"audio_bytes": b"", "audio_size_bytes": 0, "latency_ms": 0.0}
    else:
        tts_result = await run_tts(ollama_result)

    total_ms = round((time.monotonic() - t_start) * 1000, 2)
    logger.info("Pipeline END | frame_id=%s | skip_tts=%s | total=%.1fms",
                frame_id, skip_tts, total_ms)

    metadata = {
        "status": "ok",
        "frame_id": frame_id,
        "mode": mode,
        "description": ollama_result["description"],
        "user_query": stt_result["text"],
        "ocr_available": ollama_result.get("ocr_recommended", False),
        "total_latency_ms": total_ms,
        "pipeline": {
            "stt":    {"text":             stt_result["text"],
                       "latency_ms":       stt_result["latency_ms"]},
            "sam":    {"masks_found":      sam_result["masks_found"],
                       "latency_ms":       sam_result["latency_ms"]},
            "blip":   {"caption":          blip_result["caption"],
                       "latency_ms":       blip_result["latency_ms"]},
            "ollama": {"latency_ms":       ollama_result["latency_ms"]},
            "tts":    {"audio_size_bytes": tts_result["audio_size_bytes"],
                       "latency_ms":       tts_result["latency_ms"]},
        },
    }

    return PipelineResult(metadata=metadata, audio_bytes=tts_result["audio_bytes"])
