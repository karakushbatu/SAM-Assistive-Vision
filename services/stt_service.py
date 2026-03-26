# -*- coding: utf-8 -*-
"""
STT Service -- Speech-to-Text Stage
=====================================
Adapter layer for the speech recognition model.

Pipeline position: Stage 1 (optional input)
    Input : Raw audio bytes (WAV/PCM) from Android, or None if no voice input
    Output: Transcribed text (Turkish) or None

When audio_bytes is None (silent frame), returns instantly with text=None.
The downstream LLM then produces a generic scene description.

Model: faster-whisper tiny on CUDA — ~35ms warm inference, ~28s first load
"""

import asyncio
import io
import time
import random
from typing import Any

import numpy as np

from core.config import settings
from core.logger import logger


# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup via load_stt_model()
# ---------------------------------------------------------------------------

_stt_model = None


def load_stt_model() -> None:
    """
    Load faster-whisper into GPU memory.
    Call once at startup. Subsequent calls are no-ops.
    """
    global _stt_model
    if _stt_model is not None:
        return

    from faster_whisper import WhisperModel

    logger.info("Loading STT model: faster-whisper/%s on CUDA ...", settings.stt_model)
    t = time.monotonic()
    _stt_model = WhisperModel(settings.stt_model, device="cuda", compute_type="float16")
    elapsed = round((time.monotonic() - t) * 1000)
    logger.info("STT model loaded in %dms", elapsed)

    # Warm up CUDA kernels with a dummy pass (avoids 8s delay on first real call)
    dummy = np.zeros(16000, dtype=np.float32)
    list(_stt_model.transcribe(dummy, language="tr", beam_size=1)[0])
    logger.info("STT CUDA warmup complete — ready for real audio")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_stt(audio_bytes: bytes | None) -> dict[str, Any]:
    """
    Transcribe speech audio to text.

    Args:
        audio_bytes: Raw WAV bytes from Android, or None if no voice this frame.

    Returns:
        {
            "text":       str | None,   # None = no voice input
            "latency_ms": float,
        }
    """
    if audio_bytes is None:
        return {"text": None, "latency_ms": 0.0}

    if settings.mock_stt:
        return await _run_mock(audio_bytes)
    return await _run_faster_whisper(audio_bytes)


# ---------------------------------------------------------------------------
# Implementation: Mock
# ---------------------------------------------------------------------------

# Weighted toward None — most frames are silent in real usage
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


# ---------------------------------------------------------------------------
# Implementation: Real faster-whisper
# ---------------------------------------------------------------------------

async def _run_faster_whisper(audio_bytes: bytes) -> dict[str, Any]:
    """
    Transcribe using faster-whisper (CTranslate2 backend, GPU).
    Expects WAV audio bytes. Runs inference in a thread to avoid blocking.
    """
    import soundfile as sf

    if _stt_model is None:
        raise RuntimeError("STT model not loaded. Set MOCK_STT=true or call load_stt_model() at startup.")

    t = time.monotonic()
    logger.info("STT [real] | audio=%d bytes", len(audio_bytes))

    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    # Resample to 16kHz if needed (faster-whisper expects 16kHz)
    if sample_rate != 16000:
        import scipy.signal as signal
        audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sample_rate))

    def _infer():
        segments, _info = _stt_model.transcribe(
            audio_array,
            language="tr",
            beam_size=1,
            vad_filter=True,          # skip silent regions
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(s.text for s in segments).strip() or None

    text = await asyncio.to_thread(_infer)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("STT [real] done | text=%s | %.1fms", repr(text), latency)
    return {"text": text, "latency_ms": latency}
