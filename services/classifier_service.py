"""
Legacy Classifier Service
=========================
This module is intentionally kept as a compatibility shim for older scripts.
The active pipeline no longer uses CLIP/ResNet classification; SAM crops are
captioned directly by blip_service.py and synthesized by the LLM.
"""

from typing import Any

from core.logger import logger

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_classifier(sam_result: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the segmented regions from SAM output.

    Args:
        sam_result: Output of sam_service.run_sam()

    Returns:
        {
            "labels":      list[str],    # one label per mask
            "confidences": list[float],  # confidence score per label
            "latency_ms":  float,
        }
    """
    logger.warning("classifier_service is deprecated; use blip_service instead.")
    return await _run_mock(sam_result)


# ---------------------------------------------------------------------------
# Implementation: Mock (Phase 1)
# ---------------------------------------------------------------------------

import asyncio
import time
import random

OBJECT_LABELS = [
    "person", "chair", "table", "door", "window",
    "car", "bicycle", "bottle", "cup", "laptop",
    "backpack", "book", "phone", "bed", "plant",
]


async def _run_mock(sam_result: dict[str, Any]) -> dict[str, Any]:
    t = time.monotonic()
    n = sam_result["masks_found"]
    logger.info("Classifier [mock] | masks=%d", n)
    await asyncio.sleep(0.01)
    labels = random.sample(OBJECT_LABELS, min(n, len(OBJECT_LABELS)))
    confidences = [round(random.uniform(0.70, 0.99), 3) for _ in labels]
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("Classifier [mock] done | labels=%s | %.1fms", labels, latency)
    return {"labels": labels, "confidences": confidences, "latency_ms": latency}


# ---------------------------------------------------------------------------
# Implementation: CLIP (Phase 3 -- uncomment and implement)
# ---------------------------------------------------------------------------

async def _run_clip(sam_result: dict[str, Any]) -> dict[str, Any]:
    """
    Phase 3 implementation placeholder.

    TODO:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # crop each bounding box from original image, run CLIP zero-shot on each
        ...
    """
    raise NotImplementedError("CLIP not yet integrated. Set MOCK_PIPELINE=true.")
