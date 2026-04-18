# -*- coding: utf-8 -*-
"""
OCR Service -- Text Extraction Stage
===================================
Primary backend is PaddleOCR. A legacy Ollama vision OCR fallback is kept for
comparison and rollback.
"""

import asyncio
import base64
import io
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.config import settings
from core.logger import logger


_OCR_EMPTY_TEXT = "okunabilir metin yok"
_OCR_PROMPT = (
    "Extract only the clearly readable text from this image. "
    "Return plain UTF-8 text lines only. "
    "Do not explain, translate, or summarize. "
    f"If no readable text exists, return exactly: {_OCR_EMPTY_TEXT}"
)
_MOCK_OCR_TEXT = [
    "Kreon 25000\nPankreatin kapsul",
    "Yulaf Ezmesi\nProtein 13g\nLif 10g",
    "okunabilir metin yok",
]

_ocr_engine = None
_ocr_lock = asyncio.Lock()


def load_ocr_engine():
    """Load the configured OCR backend singleton."""
    global _ocr_engine
    if settings.ocr_backend != "paddleocr":
        return None
    if _ocr_engine is not None:
        return _ocr_engine

    _prepare_paddle_environment()
    from paddleocr import PaddleOCR

    preferred_device = "gpu:0" if settings.model_device == "cuda" else "cpu"
    engine = _create_paddle_engine(PaddleOCR, preferred_device, settings.paddleocr_lang)
    _ocr_engine = engine
    logger.info(
        "PaddleOCR loaded | lang=%s | version=%s | device=%s",
        settings.paddleocr_lang,
        settings.paddleocr_version,
        preferred_device,
    )
    return _ocr_engine


async def run_ocr(image_bytes: bytes, sam_result: dict[str, Any]) -> dict[str, Any]:
    """Extract printed text from the best available image region."""
    if settings.mock_ollama:
        return await _run_mock(sam_result)
    if settings.ocr_backend == "paddleocr":
        return await _run_paddle(image_bytes, sam_result)
    return await _run_ollama_vision(image_bytes, sam_result)


async def _run_mock(sam_result: dict[str, Any]) -> dict[str, Any]:
    t = time.monotonic()
    await asyncio.sleep(settings.mock_ollama_delay_ms / 1000)
    text = random.choice(_MOCK_OCR_TEXT)
    latency = round((time.monotonic() - t) * 1000, 2)
    return {
        "text": None if text == _OCR_EMPTY_TEXT else text,
        "status": "ok" if text != _OCR_EMPTY_TEXT else "empty",
        "source": "sam_crop" if sam_result.get("crops") else "full_image",
        "latency_ms": latency,
        "model": _ocr_model_name(),
    }


async def _run_paddle(image_bytes: bytes, sam_result: dict[str, Any]) -> dict[str, Any]:
    t = time.monotonic()
    crops: list[bytes] = sam_result.get("crops", [])
    candidates: list[tuple[str, bytes]] = [("sam_crop", crop) for crop in crops]
    candidates.append(("full_image", image_bytes))
    primary_source = "sam_crop" if crops else "full_image"

    logger.info(
        "OCR [paddle] | source=%s | candidates=%d",
        primary_source,
        len(candidates),
    )

    async with _ocr_lock:
        engine = await asyncio.to_thread(load_ocr_engine)
        cleaned_text = None
        used_source = primary_source
        for source, candidate_bytes in candidates:
            np_image = _decode_image(candidate_bytes)
            result = await asyncio.to_thread(
                engine.predict,
                np_image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            cleaned_text = _extract_paddle_text(result)
            used_source = source
            if cleaned_text:
                break

    status = "ok" if cleaned_text else "empty"
    latency = round((time.monotonic() - t) * 1000, 2)

    logger.info(
        "OCR [paddle] done | source=%s | status=%s | %.1fms | preview=%s",
        used_source,
        status,
        latency,
        (cleaned_text or _OCR_EMPTY_TEXT)[:80],
    )

    return {
        "text": cleaned_text,
        "status": status,
        "source": used_source,
        "latency_ms": latency,
        "model": _ocr_model_name(),
    }


async def _run_ollama_vision(image_bytes: bytes, sam_result: dict[str, Any]) -> dict[str, Any]:
    import httpx

    t = time.monotonic()
    crops: list[bytes] = sam_result.get("crops", [])
    source = "sam_crop" if crops else "full_image"
    selected_image = crops[0] if crops else image_bytes
    encoded = base64.b64encode(selected_image).decode("ascii")
    timeout = httpx.Timeout(settings.ocr_timeout_seconds, connect=3.0)

    logger.info(
        "OCR [ollama] | model=%s | source=%s | bytes=%d",
        settings.ocr_model,
        source,
        len(selected_image),
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ocr_model,
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive,
                    "messages": [
                        {
                            "role": "user",
                            "content": _OCR_PROMPT,
                            "images": [encoded],
                        }
                    ],
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 256,
                        "num_ctx": settings.ollama_num_ctx,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"OCR model is unreachable at {settings.ollama_base_url}. "
            "Make sure Ollama is running: ollama serve"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"OCR model '{settings.ocr_model}' timed out after "
            f"{settings.ocr_timeout_seconds}s."
        ) from exc

    message = payload.get("message") or {}
    raw_text = (message.get("content") or "").strip()
    cleaned_text = _clean_ocr_output(raw_text)
    status = "ok" if cleaned_text else "empty"
    latency = round((time.monotonic() - t) * 1000, 2)

    logger.info(
        "OCR [ollama] done | source=%s | status=%s | %.1fms | preview=%s",
        source,
        status,
        latency,
        (cleaned_text or _OCR_EMPTY_TEXT)[:80],
    )

    return {
        "text": cleaned_text,
        "status": status,
        "source": source,
        "latency_ms": latency,
        "model": settings.ocr_model,
    }


def _prepare_paddle_environment() -> None:
    cache_root = Path(".cache").resolve()
    paddlex_home = cache_root / "paddlex"
    paddle_home = cache_root / "paddle"

    paddlex_home.mkdir(parents=True, exist_ok=True)
    paddle_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paddlex_home))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_HOME", str(paddle_home))


def _create_paddle_engine(PaddleOCR, preferred_device: str, lang: str):
    init_kwargs = {
        "lang": lang,
        "ocr_version": settings.paddleocr_version,
        "device": preferred_device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_rec_score_thresh": 0.35,
    }
    try:
        return PaddleOCR(**init_kwargs)
    except Exception as first_exc:
        if preferred_device != "cpu":
            logger.warning(
                "PaddleOCR GPU init failed, falling back to CPU: %s",
                first_exc,
            )
            return PaddleOCR(
                lang=lang,
                ocr_version=settings.paddleocr_version,
                device="cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_rec_score_thresh=0.35,
            )
        if lang != "en":
            logger.warning(
                "PaddleOCR lang '%s' init failed, falling back to 'en': %s",
                lang,
                first_exc,
            )
            return PaddleOCR(
                lang="en",
                ocr_version=settings.paddleocr_version,
                device="cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_rec_score_thresh=0.35,
            )
        raise


def _decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except OSError as exc:
        raise RuntimeError("OCR input image could not be decoded.") from exc
    return np.array(image)


def _extract_paddle_text(predictions: Any) -> str | None:
    texts: list[str] = []
    for item in predictions or []:
        data = item
        if hasattr(item, "json"):
            try:
                data = item.json
            except Exception:
                data = item
        if isinstance(data, dict):
            rec_texts = data.get("rec_texts") or []
            rec_scores = data.get("rec_scores") or []
            for index, text in enumerate(rec_texts):
                clean_text = str(text).strip()
                if not clean_text:
                    continue
                score = None
                if index < len(rec_scores):
                    try:
                        score = float(rec_scores[index])
                    except (TypeError, ValueError):
                        score = None
                if score is not None and score < 0.35:
                    continue
                texts.append(clean_text)

    merged = "\n".join(texts).strip()
    return _clean_ocr_output(merged)


def _clean_ocr_output(raw_text: str) -> str | None:
    lines = []
    for line in raw_text.splitlines():
        clean_line = line.strip().strip("`").strip()
        if not clean_line:
            continue
        if clean_line.lower() == _OCR_EMPTY_TEXT:
            return None
        lines.append(clean_line)

    text = "\n".join(lines).strip()
    return text or None


def _ocr_model_name() -> str:
    if settings.ocr_backend == "paddleocr":
        return f"paddleocr:{settings.paddleocr_version}:{settings.paddleocr_lang}"
    return settings.ocr_model
