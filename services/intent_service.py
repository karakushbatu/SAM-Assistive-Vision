# -*- coding: utf-8 -*-
"""
Intent Service -- Voice Command Dispatcher
==========================================
Converts STT transcription text into a structured intent + pipeline mode.
No ML model needed — fast keyword matching is reliable enough for Turkish
assistive voice commands.

This service sits between STT (Stage 1) and the pipeline router in main.py.
It determines:
    - What pipeline mode to use for this frame (scene / ocr / replay / mute)
    - Whether to keep or reset conversation context

Intent categories:
    "scene"   — describe the scene (default, or explicitly requested)
    "ocr"     — read text from the object being shown
    "replay"  — repeat the last response without reprocessing
    "mute"    — suppress audio output this frame
    "camera"  — wake/open camera (Android handles this, backend just acks)
    None      — no recognized command, continue current mode
"""

from typing import Any


# ---------------------------------------------------------------------------
# Keyword maps — Turkish voice commands
# ---------------------------------------------------------------------------

_OCR_KEYWORDS = [
    "oku", "okut", "ne yazıyor", "yazıları oku", "yazıları okut",
    "etiket", "talimat", "talimatı oku", "evet", "olur", "tamam",
]

_SCENE_KEYWORDS = [
    "anlat", "ne var", "ne görüyorsun", "açıkla", "tanımla",
    "devam et", "normal mod", "sahne",
]

_REPLAY_KEYWORDS = [
    "tekrar", "bir daha", "tekrar söyle", "tekrar anlat",
]

_MUTE_KEYWORDS = [
    "dur", "yeter", "sessiz", "kapat sesi",
]

_CAMERA_KEYWORDS = [
    "kamerayı aç", "kamera aç", "başlat", "aç",
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def detect_intent(stt_text: str | None) -> dict[str, Any]:
    """
    Classify STT transcription text into a pipeline intent.

    Args:
        stt_text: Transcribed text from STT, or None if silent frame.

    Returns:
        {
            "intent":         str | None,   # "scene"|"ocr"|"replay"|"mute"|"camera"|None
            "keeps_context":  bool,         # False = reset conversation history
            "raw_text":       str | None,   # original STT text (pass-through)
        }
    """
    if not stt_text:
        return {"intent": None, "keeps_context": True, "raw_text": None}

    normalized = stt_text.lower().strip()

    if _matches(normalized, _OCR_KEYWORDS):
        return {"intent": "ocr", "keeps_context": True, "raw_text": stt_text}

    if _matches(normalized, _REPLAY_KEYWORDS):
        return {"intent": "replay", "keeps_context": True, "raw_text": stt_text}

    if _matches(normalized, _MUTE_KEYWORDS):
        return {"intent": "mute", "keeps_context": True, "raw_text": stt_text}

    if _matches(normalized, _CAMERA_KEYWORDS):
        return {"intent": "camera", "keeps_context": True, "raw_text": stt_text}

    if _matches(normalized, _SCENE_KEYWORDS):
        return {"intent": "scene", "keeps_context": True, "raw_text": stt_text}

    # No keyword matched → treat as a free-form question for the current mode
    return {"intent": None, "keeps_context": True, "raw_text": stt_text}


def should_offer_ocr(blip_caption: str) -> bool:
    """
    Check if the scene contains an object where OCR would be valuable.
    Used to add a proactive OCR offer to the Ollama prompt.

    Only medicine/prescription items and technical labels trigger this.
    Regular food products do NOT (user can still ask, but we don't proactively offer).
    """
    caption_lower = blip_caption.lower()
    ocr_trigger_keywords = [
        "medicine", "medication", "medical", "pill", "tablet", "capsule",
        "prescription", "pharmaceutical", "drug", "syringe", "inhaler",
        "bottle", "box", "label", "instruction", "warning",
    ]
    return any(kw in caption_lower for kw in ocr_trigger_keywords)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)
