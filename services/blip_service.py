# -*- coding: utf-8 -*-
"""
Caption Service (BLIP) -- Image Captioning Stage
================================================
Adapter layer for the image captioning model.

Pipeline position: Stage 4 (after SAM)
    Input : Raw image bytes + SAM segmentation result (crops)
    Output: A natural language caption describing the scene

Current model: Salesforce/blip-image-captioning-large.
Moondream2 was tested and reverted because the available Windows setup needed
native libvips or incompatible Transformers revisions.

To swap back to BLIP or try another caption model:
    - Change BLIP_MODEL in .env (and BLIP_MODEL_REVISION if needed)
    - Update load_blip_model() and _caption_pil() below
    - Function signatures and return shape must stay identical
"""

import asyncio
import time
import random
from typing import Any

from core.config import settings
from core.logger import logger
from core.runtime import gpu_inference_lock, resolve_model_device, torch_dtype_for_device


# ---------------------------------------------------------------------------
# Model singleton -- loaded once at startup, reused across all frames
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def load_blip_model() -> None:
    """
    Load BLIP processor and model into GPU memory.
    Call this once at application startup (via main.py lifespan).
    Subsequent calls are no-ops.
    """
    global _model, _tokenizer
    if _model is not None:
        return  # already loaded

    from transformers import (
        AutoTokenizer,
        BlipForConditionalGeneration,
        BlipImageProcessor,
        BlipProcessor,
    )

    device = resolve_model_device()
    torch_dtype = torch_dtype_for_device(device)
    logger.info("Loading caption model: %s ...", settings.blip_model)
    t = time.monotonic()

    try:
        _tokenizer = BlipProcessor.from_pretrained(
            settings.blip_model,
            local_files_only=settings.hf_local_files_only,
        )
    except OSError:
        logger.warning(
            "BLIP processor_config missing; falling back to image_processor + tokenizer."
        )
        image_processor = BlipImageProcessor.from_pretrained(
            settings.blip_model,
            local_files_only=settings.hf_local_files_only,
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            settings.blip_model,
            local_files_only=settings.hf_local_files_only,
        )
        _tokenizer = BlipProcessor(image_processor=image_processor, tokenizer=text_tokenizer)

    _model = BlipForConditionalGeneration.from_pretrained(
        settings.blip_model,
        torch_dtype=torch_dtype,
        local_files_only=settings.hf_local_files_only,
    ).to(device)
    _model.eval()

    elapsed = round((time.monotonic() - t) * 1000)
    logger.info("Caption model loaded in %dms | VRAM: %.1fGB",
                elapsed, _allocated_vram_gb())


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_blip(image_bytes: bytes, sam_result: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a scene caption from the image using the configured caption model.

    Args:
        image_bytes: Raw JPEG/PNG bytes (same frame passed to SAM).
        sam_result:  Output of sam_service.run_sam() -- provides crops for
                     per-object captioning.

    Returns:
        {
            "caption":    str,    # English caption (LLM stage translates to Turkish)
            "latency_ms": float,
        }
    """
    if settings.mock_blip:
        return await _run_mock(image_bytes, sam_result)
    return await _run_caption_model(image_bytes, sam_result)


# ---------------------------------------------------------------------------
# Implementation: Mock
# ---------------------------------------------------------------------------

_MOCK_CAPTIONS_EN = [
    "a person sitting at a desk with a laptop and coffee cup",
    "a hallway with a door on the left and a window at the end",
    "a street with parked cars and a bicycle leaning against a wall",
    "an office space with multiple desks, chairs, and computers",
    "a kitchen counter with various objects including bottles and utensils",
    "a living room with a sofa, coffee table, and television",
    "a bookshelf filled with books next to a potted plant",
    "a staircase going up with a railing on the right side",
]


async def _run_mock(image_bytes: bytes, sam_result: dict[str, Any]) -> dict[str, Any]:
    t = time.monotonic()
    n = sam_result["masks_found"]
    logger.info("Caption [mock] | masks=%d | image=%d bytes", n, len(image_bytes))
    await asyncio.sleep(settings.mock_blip_delay_ms / 1000)
    caption = random.choice(_MOCK_CAPTIONS_EN)
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("Caption [mock] done | %.1fms | caption=%s", latency, caption)
    return {"caption": caption, "latency_ms": latency}


# ---------------------------------------------------------------------------
# Implementation: Real BLIP model
# ---------------------------------------------------------------------------

async def _run_caption_model(image_bytes: bytes, sam_result: dict[str, Any]) -> dict[str, Any]:
    """
    Run BLIP inference.

    If SAM produced crops, caption each crop separately and join them.
    This gives the LLM richer per-object descriptions than a single
    whole-image caption. Falls back to full image if no crops available.
    """
    import io
    from PIL import Image

    if _model is None:
        raise RuntimeError(
            "Caption model not loaded. Set MOCK_BLIP=true or call load_blip_model() at startup."
        )

    t = time.monotonic()
    crops: list[bytes] = sam_result.get("crops", [])
    n = sam_result["masks_found"]
    logger.info("Caption [real] | masks=%d | crops=%d | image=%d bytes",
                n, len(crops), len(image_bytes))

    def _caption_pil(pil_img) -> str:
        """Caption a single PIL image using BLIP."""
        import torch
        device = resolve_model_device()
        dtype = torch_dtype_for_device(device)
        inputs = _tokenizer(pil_img, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=settings.blip_max_new_tokens)
        return _tokenizer.decode(out[0], skip_special_tokens=True)

    def _open_full_image() -> Image.Image:
        """Open the full frame, resize if oversized."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(img.size) > settings.max_frame_size:
            img.thumbnail((settings.max_frame_size, settings.max_frame_size))
        return img

    def _infer():
        if crops:
            import torch
            # Batch all crops into a single GPU forward pass (faster than sequential)
            device = resolve_model_device()
            dtype = torch_dtype_for_device(device)
            crop_imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in crops]
            inputs = _tokenizer(crop_imgs, return_tensors="pt", padding=True).to(device, dtype)
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=settings.blip_max_new_tokens)
            raw_captions = [_tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
            # Filter artifacts and duplicates
            seen: set[str] = set()
            captions = []
            for cap in raw_captions:
                if cap and cap not in seen and "araf" not in cap.lower():
                    seen.add(cap)
                    captions.append(cap)
            if captions:
                return "; ".join(captions[:settings.blip_max_captions])[:200]
            # All crops produced bad output -- fall back to full image
            return _caption_pil(_open_full_image())
        else:
            # No SAM crops -- fall back to full image
            return _caption_pil(_open_full_image())

    async with gpu_inference_lock:
        caption = await asyncio.to_thread(_infer)

    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("Caption [real] done | %.1fms | caption=%s", latency, caption)
    return {"caption": caption, "latency_ms": latency}


def _allocated_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0
