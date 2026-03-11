"""
AI Pipeline -- Orchestrator
============================
This file ONLY coordinates the pipeline stages. It contains NO model logic.

Each stage is handled by its own service module:
    sam_service.py        -> segmentation
    classifier_service.py -> object labeling
    ollama_service.py     -> Turkish description (LLM)
    tts_service.py        -> speech audio (TTS)

To swap any model/engine: edit the corresponding service file.
To change pipeline order or add a stage: edit only this file.
"""

import time
import uuid
from typing import Any

from core.logger import logger
from services.sam_service import run_sam
from services.classifier_service import run_classifier
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


async def run_full_pipeline(image_bytes: bytes) -> PipelineResult:
    """
    Execute the 4-stage pipeline sequentially on one image frame.

    Stage order:
        1. SAM           (sam_service)         -- segments image into regions
        2. Classifier    (classifier_service)  -- labels each region
        3. Ollama        (ollama_service)       -- generates Turkish description
        4. TTS           (tts_service)          -- converts text to MP3 audio

    Args:
        image_bytes: Raw JPEG/PNG bytes from the Android WebSocket client.

    Returns:
        PipelineResult:
            .metadata    -> JSON-serializable dict with description + latencies
            .audio_bytes -> MP3 bytes to play on the Android device
    """
    frame_id = str(uuid.uuid4())
    t_start = time.monotonic()
    logger.info("Pipeline START | frame_id=%s", frame_id)

    sam_result        = await run_sam(image_bytes)
    classifier_result = await run_classifier(sam_result)
    ollama_result     = await run_ollama(classifier_result)
    tts_result        = await run_tts(ollama_result)

    total_ms = round((time.monotonic() - t_start) * 1000, 2)
    logger.info("Pipeline END | frame_id=%s | total=%.1fms", frame_id, total_ms)

    metadata = {
        "status": "ok",
        "frame_id": frame_id,
        "description": ollama_result["description"],
        "total_latency_ms": total_ms,
        "pipeline": {
            "sam":        {"masks_found": sam_result["masks_found"],
                           "latency_ms":  sam_result["latency_ms"]},
            "classifier": {"labels":      classifier_result["labels"],
                           "latency_ms":  classifier_result["latency_ms"]},
            "ollama":     {"latency_ms":  ollama_result["latency_ms"]},
            "tts":        {"audio_size_bytes": tts_result["audio_size_bytes"],
                           "latency_ms":       tts_result["latency_ms"]},
        },
    }

    return PipelineResult(metadata=metadata, audio_bytes=tts_result["audio_bytes"])
