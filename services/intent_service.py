# -*- coding: utf-8 -*-
"""
Intent Service -- Turkish Voice Command Dispatcher
==================================================
Fast keyword routing for assistive voice commands.
"""

import re
import unicodedata
from typing import Any


_OCR_KEYWORDS = [
    "oku",
    "okut",
    "ne yaziyor",
    "yazilari oku",
    "etiketi oku",
    "tabelayi oku",
]

_SCENE_KEYWORDS = [
    "anlat",
    "ne var",
    "ne goruyorsun",
    "acikla",
    "tanimla",
    "normal mod",
    "sahne",
]

_REPLAY_KEYWORDS = [
    "tekrar",
    "bir daha",
    "tekrar soyle",
    "tekrar anlat",
]

_MUTE_KEYWORDS = [
    "dur",
    "yeter",
    "sessiz",
    "kapat sesi",
]

_CAMERA_KEYWORDS = [
    "kamerayi ac",
    "kamera ac",
    "baslat",
]


def detect_intent(stt_text: str | None) -> dict[str, Any]:
    if not stt_text:
        return {"intent": None, "keeps_context": True, "raw_text": None}

    normalized = _normalize(stt_text)

    if _exact_match(normalized, _REPLAY_KEYWORDS):
        return {"intent": "replay", "keeps_context": True, "raw_text": stt_text}
    if _exact_match(normalized, _MUTE_KEYWORDS):
        return {"intent": "mute", "keeps_context": True, "raw_text": stt_text}
    if _exact_match(normalized, _CAMERA_KEYWORDS):
        return {"intent": "camera", "keeps_context": True, "raw_text": stt_text}
    if _matches(normalized, _OCR_KEYWORDS):
        return {"intent": "ocr", "keeps_context": True, "raw_text": stt_text}
    if _matches(normalized, _SCENE_KEYWORDS):
        return {"intent": "scene", "keeps_context": True, "raw_text": stt_text}

    return {"intent": None, "keeps_context": True, "raw_text": stt_text}


def should_offer_ocr(blip_caption: str) -> bool:
    caption_lower = (blip_caption or "").lower()
    ocr_trigger_keywords = [
        "medicine",
        "medication",
        "medical",
        "pill",
        "pills",
        "tablet",
        "capsule",
        "prescription",
        "pharmaceutical",
        "drug",
        "syringe",
        "inhaler",
        "bottle",
        "box",
        "label",
        "instruction",
        "warning",
        "receipt",
        "menu",
        "sign",
        "document",
        "nutrition",
        "ingredients",
    ]
    return any(keyword in caption_lower for keyword in ocr_trigger_keywords)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = (
        text.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ş", "s")
        .replace("ç", "c")
        .replace("ö", "o")
        .replace("ü", "u")
    )
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _matches(text: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        normalized_keyword = _normalize(keyword)
        pattern = rf"(?<!\w){re.escape(normalized_keyword)}(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _exact_match(text: str, keywords: list[str]) -> bool:
    return any(text == _normalize(keyword) for keyword in keywords)
