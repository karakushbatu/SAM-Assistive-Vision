# -*- coding: utf-8 -*-
"""
Ollama Service -- Turkish Response Layer
========================================
This stage converts vision/OCR outputs into short Turkish responses suitable
for low-latency assistive audio feedback.
"""

import asyncio
import random
import re
import time
from typing import Any

from core.config import settings
from core.logger import logger


_SCENE_SYSTEM_PROMPT = (
    "Sen görme engelli kullanıcılar için çalışan bir yardımcı asistansın. "
    "Sadece kısa ve doğal Türkçe yaz. "
    "Cevap en fazla 2 cümle olsun. "
    "En önemli nesneyi, engeli, yön bilgisini veya kullanıcının sorduğu şeyi söyle. "
    "İngilizce sahne ipuçlarını doğru Türkçeye çevir. "
    "Şu çevirilere dikkat et: pill veya pills=ilaç hapı, box of pills=ilaç kutusu, "
    "oatmeal=yulaf ezmesi, spoon full of oatmeal=kaşıkta yulaf ezmesi, "
    "cereal=kahvaltılık gevrek, bag=paket, gun=silah, game=oyun ekranı, "
    "person=kişi, close up=yakın çekim. "
    "Marka veya ilaç adı açıkça görünüyorsa adını koru. "
    "Asla 'individü' gibi yabancılaşmış kelimeler kullanma. "
    "Uydurma nesne, renk veya arka plan ekleme."
)

_OCR_SYSTEM_PROMPT = (
    "Sen görme engelli kullanıcılar için çalışan bir yardımcı asistansın. "
    "Sadece Türkçe yaz. "
    "Sana görüntüden çıkarılmış ham metin verilecek. "
    "Metindeki en önemli bilgiyi 1-2 kısa cümlede özetle. "
    "İlaç için ad, tür ve gerekiyorsa doz; gıda için ürün ve önemli besin bilgisi; "
    "etiket veya tabela için doğrudan işe yarayan ana bilgiyi söyle. "
    "Yulaf, gevrek, protein, lif, içerik gibi kelimeleri ilaç olarak yorumlama; bunlar gıda ürünüdür. "
    "Eksik veya belirsiz kısım varsa bunu kısa biçimde belirt."
)

_USER_PROMPT_SCENE = (
    "Sahne ipuçları:\n{caption}\n\n"
    "Kullanıcı sorusu: {user_query}\n"
    "Yalnızca Türkçe cevap ver:"
)

_USER_PROMPT_OCR = (
    "OCR metni:\n{ocr_text}\n\n"
    "Kullanıcı sorusu: {user_query}\n"
    "Yalnızca Türkçe cevap ver:"
)

_NO_TEXT_RESPONSE = "Okunabilir bir metin tespit edemedim."

_MOCK_DESCRIPTIONS_TR = [
    "Onunuzde bir kutu var, arkasinda baska bir nesne gorunuyor.",
    "Masanin ustunde bir bilgisayar ve bardak var.",
    "Solunuzda bir araba, saginizda bir bisiklet gorunuyor.",
]

_MOCK_OCR_RESPONSES_TR = [
    "Bu bir ilac kutusu. Uzerinde Kreon 25000 yaziyor.",
    "Paketin uzerinde yulaf ezmesi ve protein bilgisi yaziyor.",
    _NO_TEXT_RESPONSE,
]


async def run_ollama(
    blip_result: dict[str, Any],
    user_query: str | None = None,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    if settings.mock_ollama:
        return await _run_mock_scene(user_query)
    caption = blip_result["caption"]
    prompt = _USER_PROMPT_SCENE.format(
        caption=caption,
        user_query=user_query or "yok",
    )
    return await _run_real(
        model=settings.scene_ollama_model,
        system_prompt=_SCENE_SYSTEM_PROMPT,
        user_prompt=prompt,
        history=history or [],
        num_predict=settings.scene_ollama_num_predict,
        num_ctx=settings.scene_ollama_num_ctx,
    )


async def run_ollama_ocr(
    ocr_result: dict[str, Any],
    user_query: str | None = None,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    if settings.mock_ollama:
        return await _run_mock_ocr()

    ocr_text = ocr_result.get("text")
    if not ocr_text:
        return {
            "description": _NO_TEXT_RESPONSE,
            "ocr_recommended": False,
            "latency_ms": 0.0,
        }

    prompt = _USER_PROMPT_OCR.format(
        ocr_text=ocr_text,
        user_query=user_query or "yok",
    )
    return await _run_real(
        model=settings.ocr_summary_model,
        system_prompt=_OCR_SYSTEM_PROMPT,
        user_prompt=prompt,
        history=history or [],
        num_predict=settings.ocr_summary_num_predict,
        num_ctx=settings.ocr_summary_num_ctx,
    )


async def _run_mock_scene(user_query: str | None) -> dict[str, Any]:
    t = time.monotonic()
    await asyncio.sleep(settings.mock_ollama_delay_ms / 1000)
    description = random.choice(_MOCK_DESCRIPTIONS_TR)
    if user_query:
        description = f"Sorunuza gore: {description}"
    latency = round((time.monotonic() - t) * 1000, 2)
    return {"description": description, "ocr_recommended": False, "latency_ms": latency}


async def _run_mock_ocr() -> dict[str, Any]:
    t = time.monotonic()
    await asyncio.sleep(settings.mock_ollama_delay_ms / 1000)
    latency = round((time.monotonic() - t) * 1000, 2)
    return {
        "description": random.choice(_MOCK_OCR_RESPONSES_TR),
        "ocr_recommended": False,
        "latency_ms": latency,
    }


async def _run_real(
    model: str,
    system_prompt: str,
    user_prompt: str,
    history: list[dict],
    num_predict: int,
    num_ctx: int,
) -> dict[str, Any]:
    import httpx

    t = time.monotonic()
    timeout = httpx.Timeout(settings.ollama_timeout_seconds, connect=3.0)
    history_text = _history_text(history)
    prompt = history_text + user_prompt

    logger.info(
        "Ollama [real] | model=%s | prompt_chars=%d",
        model,
        len(prompt),
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "system": system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive,
                    "options": {
                        "temperature": settings.ollama_temperature,
                        "num_predict": num_predict,
                        "num_ctx": num_ctx,
                    },
                },
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Ollama is unreachable at {settings.ollama_base_url}. "
            "Make sure Ollama is running: ollama serve"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Ollama timed out after {settings.ollama_timeout_seconds}s. "
            f"Model '{model}' may still be loading."
        ) from exc

    description = _clean_output(raw_text)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("Ollama [real] done | %.1fms | output=%s", latency, description[:80])
    return {"description": description, "ocr_recommended": False, "latency_ms": latency}


def _history_text(history: list[dict]) -> str:
    snippets = [item["assistant"] for item in history[-1:] if item.get("assistant")]
    if not snippets:
        return ""
    return "Baglam: " + " ".join(snippets) + "\n\n"


def _clean_output(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^[^\n]{0,40}:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip(" \"'")
    return text or _NO_TEXT_RESPONSE
