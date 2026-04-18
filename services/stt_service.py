# -*- coding: utf-8 -*-
"""
STT Service -- Speech-to-Text Stage
===================================
Adapter layer for Turkish speech recognition.

Input is optional WAV audio bytes from the Android WebSocket envelope. Silent
image-only frames pass None and return immediately.
"""

import asyncio
import io
import random
import time
from typing import Any

import numpy as np

from core.config import settings
from core.logger import logger
from core.runtime import resolve_model_device


_stt_model = None


def load_stt_model() -> None:
    """Load faster-whisper into memory. Subsequent calls are no-ops."""
    global _stt_model
    if _stt_model is not None:
        return

    from faster_whisper import WhisperModel

    device = resolve_model_device()
    compute_type = settings.stt_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "Loading STT model: faster-whisper/%s on %s (%s) ...",
        settings.stt_model,
        device,
        compute_type,
    )
    t = time.monotonic()
    _stt_model = WhisperModel(
        settings.stt_model,
        device=device,
        compute_type=compute_type,
        local_files_only=settings.hf_local_files_only,
    )
    elapsed = round((time.monotonic() - t) * 1000)
    logger.info("STT model loaded in %dms", elapsed)

    dummy = np.zeros(16000, dtype=np.float32)
    list(_stt_model.transcribe(dummy, language="tr", beam_size=1)[0])
    logger.info("STT warmup complete")


async def run_stt(audio_bytes: bytes | None) -> dict[str, Any]:
    """
    Transcribe speech audio to Turkish text.
    """
    if audio_bytes is None:
        return {"text": None, "latency_ms": 0.0}

    if settings.mock_stt:
        return await _run_mock(audio_bytes)
    return await _run_faster_whisper(audio_bytes)


_MOCK_QUERIES_TR = [
    "Önümde ne var?",
    "Bu nesne nedir?",
    "Yolumda engel var mı?",
    "Kapı nerede?",
    "Bunu oku.",
    None,
    None,
    None,
]


async def _run_mock(audio_bytes: bytes) -> dict[str, Any]:
    t = time.monotonic()
    logger.info("STT [mock] | audio=%d bytes", len(audio_bytes))
    await asyncio.sleep(settings.mock_stt_delay_ms / 1000)
    text = random.choice(_MOCK_QUERIES_TR)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("STT [mock] done | text=%s | %.1fms", repr(text), latency)
    return {"text": text, "latency_ms": latency}


async def _run_faster_whisper(audio_bytes: bytes) -> dict[str, Any]:
    """
    Transcribe using faster-whisper. Audio parsing, resampling, and inference
    run in a worker thread so malformed or long audio does not block the event loop.
    """
    if _stt_model is None:
        raise RuntimeError("STT model not loaded. Set MOCK_STT=true or call load_stt_model() at startup.")

    t = time.monotonic()
    logger.info("STT [real] | audio=%d bytes", len(audio_bytes))

    def _infer() -> str | None:
        import soundfile as sf

        try:
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            raise ValueError(f"Invalid audio payload: {exc}") from exc

        if audio_array.size == 0:
            return None

        duration_seconds = len(audio_array) / float(sample_rate)
        if duration_seconds > settings.stt_max_audio_seconds:
            raise ValueError(
                f"Audio too long: {duration_seconds:.1f}s exceeds "
                f"STT_MAX_AUDIO_SECONDS={settings.stt_max_audio_seconds}"
            )

        if sample_rate != 16000:
            import scipy.signal as signal
            audio_array = signal.resample(
                audio_array,
                int(len(audio_array) * 16000 / sample_rate),
            )

        segments, _info = _stt_model.transcribe(
            audio_array,
            language="tr",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(s.text for s in segments).strip() or None

    text = await asyncio.to_thread(_infer)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("STT [real] done | text=%s | %.1fms", repr(text), latency)
    return {"text": text, "latency_ms": latency}
