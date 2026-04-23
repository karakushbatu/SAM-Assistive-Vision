"""
TTS Service -- Text-to-Speech Stage
=====================================
Adapter layer for the text-to-speech engine.

To swap the TTS engine (Edge TTS <-> Coqui <-> ElevenLabs <-> any future):
    - Only change _run_real() below
    - Voice/speed are controlled via .env (TTS_VOICE, TTS_RATE)
    - The function signature and return shape MUST stay identical

Current state: MOCK (Phase 1)
Next: Edge TTS -- Microsoft Neural Voices, free, no API key (Phase 4)

Available Turkish voices (Edge TTS):
    tr-TR-EmelNeural    -- female, clear, natural
    tr-TR-AhmetNeural   -- male
"""

from typing import Any, AsyncGenerator

from core.config import settings
from core.logger import logger

# Minimal silent MP3 frame (placeholder in mock mode)
_SILENT_MP3 = bytes([
    0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_tts(ollama_result: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Turkish text description into MP3 audio bytes.

    Args:
        ollama_result: Output of ollama_service.run_ollama()

    Returns:
        {
            "audio_bytes":      bytes,
            "audio_size_bytes": int,
            "latency_ms":       float,
        }
    """
    if settings.mock_tts:
        return await _run_mock(ollama_result)
    try:
        return await _run_edge_tts(ollama_result)
    except Exception as exc:
        logger.warning("TTS [edge] failed, falling back to silent audio: %s", exc)
        fallback = await _run_mock(ollama_result)
        fallback["fallback"] = "silent_mock"
        fallback["error"] = str(exc)
        return fallback


# ---------------------------------------------------------------------------
# Implementation: Mock (Phase 1)
# ---------------------------------------------------------------------------

import asyncio
import time


async def _run_mock(ollama_result: dict[str, Any]) -> dict[str, Any]:
    t = time.monotonic()
    text = ollama_result["description"]
    logger.info("TTS [mock] | voice=%s | len=%d chars", settings.tts_voice, len(text))
    await asyncio.sleep(settings.mock_tts_delay_ms / 1000)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("TTS [mock] done | %.1fms", latency)
    return {"audio_bytes": _SILENT_MP3, "audio_size_bytes": len(_SILENT_MP3), "latency_ms": latency}


# ---------------------------------------------------------------------------
# Implementation: Edge TTS (Phase 4)
# ---------------------------------------------------------------------------

async def _run_edge_tts(ollama_result: dict[str, Any]) -> dict[str, Any]:
    """
    Phase 4 implementation using Microsoft Edge TTS.
    Free, no API key, excellent Turkish neural voices.

    Install: pip install edge-tts
    """
    import io
    import edge_tts  # pip install edge-tts

    t = time.monotonic()
    text = ollama_result["description"]
    logger.info("TTS [edge] | voice=%s | rate=%s", settings.tts_voice, settings.tts_rate)

    communicate = edge_tts.Communicate(text, settings.tts_voice, rate=settings.tts_rate)

    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])

    audio_bytes = buffer.getvalue()
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("TTS [edge] done | size=%d bytes | %.1fms", len(audio_bytes), latency)
    return {"audio_bytes": audio_bytes, "audio_size_bytes": len(audio_bytes), "latency_ms": latency}


# ---------------------------------------------------------------------------
# Streaming interface (TTS_STREAMING=true)
# ---------------------------------------------------------------------------

async def stream_tts(text: str) -> AsyncGenerator[bytes, None]:
    """
    Yield MP3 audio chunks as Edge TTS generates them.

    Allows the WebSocket handler to send audio to the client before the
    full synthesis is complete — reduces perceived latency.
    Only used when TTS_STREAMING=true and MOCK_TTS=false.

    Args:
        text: Turkish description text to synthesize.

    Yields:
        bytes: MP3 audio chunk.
    """
    import edge_tts
    communicate = edge_tts.Communicate(text, settings.tts_voice, rate=settings.tts_rate)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


def get_silent_audio_bytes() -> bytes:
    return _SILENT_MP3


# ---------------------------------------------------------------------------
# Implementation stub: Alternative engines (future)
# ---------------------------------------------------------------------------

async def _run_coqui_tts(ollama_result: dict[str, Any]) -> dict[str, Any]:
    """
    Future: Coqui TTS (open-source, can run fully offline on GPU).
    Useful if Edge TTS quality is not sufficient or offline mode is needed.
    Install: pip install TTS
    """
    raise NotImplementedError("Coqui TTS not yet integrated.")


async def _run_elevenlabs(ollama_result: dict[str, Any]) -> dict[str, Any]:
    """
    Future: ElevenLabs API (highest quality, paid, requires API key).
    Useful for production-grade voice quality.
    """
    raise NotImplementedError("ElevenLabs not yet integrated.")
