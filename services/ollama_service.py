# -*- coding: utf-8 -*-
"""
Ollama Service -- LLM Description Stage
=========================================
Adapter layer for the language model.

To swap the model (llama3.2:3b <-> mistral <-> gemma2 <-> any future model):
    - Change OLLAMA_MODEL in your .env file -- that is ALL, no code change needed.

To swap the LLM provider entirely (Ollama <-> OpenAI <-> local llama.cpp):
    - Only change _run_real() below.
    - Function signature and return shape must stay identical.

Turkish output:
    System prompt uses proper Turkish characters (ş, ı, ğ, ü, ö, ç).
    This produces cleaner TTS output from Edge TTS neural voices.
"""

import asyncio
import re
import time
import random
from typing import Any

from core.config import settings
from core.logger import logger

# ---------------------------------------------------------------------------
# Prompt templates — proper Turkish characters for clean TTS output
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Sen görme engelli kullanıcılara yardım eden bir görsel asistansın. "
    "Görev: Sana verilen nesne listesinden yola çıkarak sahneyi Türkçe olarak "
    "SADECE 1-2 düz cümleyle açıkla. "
    "Önemli kurallar: "
    "1) Asla madde işareti, tire veya liste kullanma. "
    "2) Asla İngilizce kelime yazma, her şeyi Türkçe yaz. "
    "3) 'önünüzde', 'solunuzda', 'sağınızda', 'arkanızda' gibi yön ifadeleri kullan. "
    "4) Cevap maksimum 2 cümle olmalı, fazlası olmasın. "
    "Örnek doğru cevap: 'Önünüzde bir masa ve sandalye var, solunuzda kapı bulunuyor.'"
)

USER_PROMPT_TEMPLATE = "Sahnedeki nesneler: {labels}. Türkçe 1-2 cümleyle açıkla."

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_ollama(classifier_result: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a Turkish scene description from classified object labels.

    Args:
        classifier_result: Output of classifier_service.run_classifier()

    Returns:
        { "description": str, "latency_ms": float }
    """
    if settings.mock_ollama:
        return await _run_mock(classifier_result)
    return await _run_real(classifier_result)


# ---------------------------------------------------------------------------
# Implementation: Mock
# ---------------------------------------------------------------------------

MOCK_DESCRIPTIONS_TR = [
    "Önünüzde bir kapı var, solunuzda bir sandalye bulunuyor.",
    "Masanın üzerinde dizüstü bilgisayar ve bardak var, sağınızda pencere görünüyor.",
    "Duvara yaslanmış bir bisiklet var, bir kişi size doğru geliyor.",
    "Bir ofis ortamındasınız, etrafta masalar ve sandalyeler var.",
    "Solunuzda bir araba park edilmiş, karşıda bina girişi görünüyor.",
]


async def _run_mock(classifier_result: dict[str, Any]) -> dict[str, Any]:
    t = time.monotonic()
    logger.info("Ollama [mock] | labels=%s", classifier_result["labels"])
    await asyncio.sleep(settings.mock_ollama_delay_ms / 1000)
    description = random.choice(MOCK_DESCRIPTIONS_TR)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("Ollama [mock] done | %.1fms", latency)
    return {"description": description, "latency_ms": latency}


# ---------------------------------------------------------------------------
# Implementation: Real Ollama HTTP API
# ---------------------------------------------------------------------------

async def _run_real(classifier_result: dict[str, Any]) -> dict[str, Any]:
    """
    Calls the Ollama HTTP API at settings.ollama_base_url.
    Model is set via OLLAMA_MODEL in .env — no code change needed to swap models.
    """
    import httpx

    t = time.monotonic()
    labels = classifier_result["labels"]
    user_prompt = USER_PROMPT_TEMPLATE.format(labels=", ".join(labels))

    payload = {
        "model": settings.ollama_model,
        "prompt": user_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 120},
    }

    logger.info("Ollama [real] | model=%s | labels=%s", settings.ollama_model, labels)

    async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    description = _clean_output(data.get("response", ""))
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("Ollama [real] done | %.1fms | output=%s", latency, description[:80])
    return {"description": description, "latency_ms": latency}


# ---------------------------------------------------------------------------
# Output cleanup — strips LLM formatting artifacts for clean TTS input
# ---------------------------------------------------------------------------

def _clean_output(raw: str) -> str:
    """
    Remove common LLM preamble and formatting so Edge TTS reads clean sentences.
    Examples removed:
        "Sahneyi açıklamak için:\n- ..."  -> strips header + bullets
        "sandalye (chair)"                -> strips English in parens
        '"Önünüzde..."'                   -> strips surrounding quotes
    """
    text = raw.strip()
    # Remove header lines ending with colon (e.g. "Açıklama:", "Sahne:")
    text = re.sub(r'^[^\n]{0,60}:\s*\n', '', text, flags=re.IGNORECASE)
    # Remove bullet/dash list markers at line start
    text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)
    # Remove English words in parentheses
    text = re.sub(r'\s*\([a-zA-Z ]+\)', '', text)
    # Strip surrounding quotes
    text = text.strip('"\'')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
