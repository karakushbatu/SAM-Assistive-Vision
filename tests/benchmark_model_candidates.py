# -*- coding: utf-8 -*-
r"""
Benchmark local LLM/OCR candidates without changing the main backend config.

Examples:
    .\.venv\Scripts\python tests\benchmark_model_candidates.py
    .\.venv\Scripts\python tests\benchmark_model_candidates.py --scene-models qwen2.5:7b llama3.2:3b gemma3:4b
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import settings
from services.sam_service import load_sam_model, run_sam


SCENE_CAPTION = "spoon full of oatmeal with a bag of cereal in the background"
SCENE_PROMPT = (
    "Sahne ipuçları:\n"
    f"{SCENE_CAPTION}\n\n"
    "Kullanıcı sorusu: yok\n"
    "Yalnızca Türkçe cevap ver:"
)

OCR_TEXT = "Abbott\nKreon\n10000\nPankreas Enzim Konsantresi\n100 Kapsül"
OCR_SUMMARY_PROMPT = (
    "OCR metni:\n"
    f"{OCR_TEXT}\n\n"
    "Kullanıcı sorusu: yok\n"
    "Yalnızca Türkçe cevap ver:"
)

SCENE_SYSTEM = (
    "Sen görme engelli kullanıcılar için çalışan bir yardımcı asistansın. "
    "Sadece kısa ve doğal Türkçe yaz. Cevap en fazla 2 cümle olsun."
)

OCR_SUMMARY_SYSTEM = (
    "Sen görme engelli kullanıcılar için çalışan bir yardımcı asistansın. "
    "Sadece Türkçe yaz. OCR metnini 1-2 kısa cümlede özetle."
)

OCR_VISION_PROMPT = (
    "Extract only the clearly readable text from this image. "
    "Return plain text lines only. If no readable text exists, return exactly: okunabilir metin yok"
)


async def benchmark_generate(
    client: httpx.AsyncClient,
    model: str,
    system: str,
    prompt: str,
    num_predict: int = 64,
    num_ctx: int = 512,
) -> dict:
    t = time.monotonic()
    response = await client.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        },
    )
    response.raise_for_status()
    data = response.json()
    return {
        "model": model,
        "latency_ms": round((time.monotonic() - t) * 1000, 2),
        "output": (data.get("response") or "").strip(),
    }


async def benchmark_vision_ocr(client: httpx.AsyncClient, model: str, image_path: str) -> dict:
    try:
        with open(image_path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise RuntimeError(f"OCR benchmark image could not be read: {image_path}") from exc
    await asyncio.to_thread(load_sam_model)
    sam_result = await run_sam(raw)
    crops = sam_result.get("crops") or []
    raw = crops[0] if crops else raw
    t = time.monotonic()
    response = await client.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": OCR_VISION_PROMPT,
                    "images": [base64.b64encode(raw).decode("ascii")],
                }
            ],
            "options": {"temperature": 0.0, "num_predict": 256, "num_ctx": 512},
        },
    )
    response.raise_for_status()
    data = response.json()
    output = ((data.get("message") or {}).get("content") or "").strip()
    return {
        "model": model,
        "latency_ms": round((time.monotonic() - t) * 1000, 2),
        "output": output,
    }


async def main(args):
    timeout = httpx.Timeout(120.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        print("=== Scene Summary Models ===")
        for model in args.scene_models:
            result = await benchmark_generate(client, model, SCENE_SYSTEM, SCENE_PROMPT)
            print(json.dumps(result, ensure_ascii=False))

        print("\n=== OCR Summary Models ===")
        for model in args.ocr_summary_models:
            result = await benchmark_generate(client, model, OCR_SUMMARY_SYSTEM, OCR_SUMMARY_PROMPT, num_predict=72)
            print(json.dumps(result, ensure_ascii=False))

        if args.ocr_model and args.ocr_image:
            print("\n=== OCR Vision Model ===")
            result = await benchmark_vision_ocr(client, args.ocr_model, args.ocr_image)
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark local model candidates")
    parser.add_argument(
        "--scene-models",
        nargs="+",
        default=["qwen2.5:7b", "qwen3:4b", "gemma3:4b", "llama3.2:3b"],
    )
    parser.add_argument(
        "--ocr-summary-models",
        nargs="+",
        default=["qwen2.5:7b", "qwen3:4b", "gemma3:4b", "llama3.2:3b"],
    )
    parser.add_argument("--ocr-model", default="glm-ocr:latest")
    parser.add_argument("--ocr-image", default="tests/sample_images/ilaç.jpg")
    args = parser.parse_args()
    asyncio.run(main(args))
