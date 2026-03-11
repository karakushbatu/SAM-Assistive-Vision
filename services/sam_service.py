"""
SAM Service -- Segmentation Stage
==================================
Adapter layer for the segmentation model.

To swap the model (FastSAM <-> MobileSAM <-> any future model):
    - Only change the implementation inside run_sam()
    - The function signature and return shape MUST stay identical
    - ai_pipeline.py (the orchestrator) never needs to change

Current state: MOCK (Phase 1)
Next: FastSAM via ultralytics (Phase 2)
"""

from typing import Any

from core.config import settings
from core.logger import logger

# ---------------------------------------------------------------------------
# Public interface -- this is the ONLY function ai_pipeline.py should call
# ---------------------------------------------------------------------------

async def run_sam(image_bytes: bytes) -> dict[str, Any]:
    """
    Run segmentation on a raw image frame.

    Args:
        image_bytes: Raw JPEG/PNG bytes from the WebSocket client.

    Returns:
        {
            "masks_found":    int,
            "bounding_boxes": list[dict],   # [{"x","y","w","h"}, ...]
            "latency_ms":     float,
        }
    """
    if settings.mock_sam:
        return await _run_mock(image_bytes)
    return await _run_fastsam(image_bytes)


# ---------------------------------------------------------------------------
# Implementation: Mock (Phase 1)
# ---------------------------------------------------------------------------

import asyncio
import time
import random


async def _run_mock(image_bytes: bytes) -> dict[str, Any]:
    t = time.monotonic()
    logger.info("SAM [mock] | input=%d bytes", len(image_bytes))
    await asyncio.sleep(settings.mock_sam_delay_ms / 1000)
    n = random.randint(2, 6)
    boxes = [{"x": random.randint(0, 300), "y": random.randint(0, 300),
               "w": random.randint(50, 200), "h": random.randint(50, 200)}
              for _ in range(n)]
    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("SAM [mock] done | masks=%d | %.1fms", n, latency)
    return {"masks_found": n, "bounding_boxes": boxes, "latency_ms": latency}


# ---------------------------------------------------------------------------
# Implementation: FastSAM (Phase 2 -- uncomment and implement)
# ---------------------------------------------------------------------------

async def _run_fastsam(image_bytes: bytes) -> dict[str, Any]:
    """
    Phase 2 implementation placeholder.

    TODO:
        import io
        from PIL import Image
        from ultralytics import FastSAM

        model = FastSAM("FastSAM-s.pt")   # load once at startup, not here
        image = Image.open(io.BytesIO(image_bytes))
        results = model(image, device="cuda", retina_masks=True, conf=0.4, iou=0.9)
        ...
    """
    raise NotImplementedError("FastSAM not yet integrated. Set MOCK_PIPELINE=true.")
