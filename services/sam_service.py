"""
SAM Service -- Segmentation Stage
==================================
Adapter layer for FastSAM segmentation model.

Input : JPEG/PNG bytes from WebSocket client
Output: bounding boxes of detected object regions

These bounding boxes are passed to blip_service which crops each region
and generates a separate caption per object — giving the LLM much richer
context than a single whole-image caption.

Model: FastSAM-s (ultralytics) — GPU-optimized, ~22MB weights
"""

import asyncio
import io
import time
import random
from typing import Any

from core.config import settings
from core.logger import logger


# ---------------------------------------------------------------------------
# B2: Frame validation — accepted image magic bytes
# ---------------------------------------------------------------------------

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC  = b"\x89PNG"


def _validate_image(image_bytes: bytes) -> None:
    """Raise ValueError if bytes are not a valid JPEG or PNG."""
    if not (image_bytes[:3] == _JPEG_MAGIC or image_bytes[:4] == _PNG_MAGIC):
        raise ValueError(
            f"Invalid image format: expected JPEG or PNG, "
            f"got magic bytes 0x{image_bytes[:4].hex()} "
            f"(size={len(image_bytes)} bytes)"
        )


# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup
# ---------------------------------------------------------------------------

_sam_model = None


def load_sam_model() -> None:
    """Load FastSAM-s into GPU memory. Call once at startup. Subsequent calls are no-ops."""
    global _sam_model
    if _sam_model is not None:
        return

    from ultralytics import FastSAM

    model_path = settings.sam_model  # "FastSAM-s.pt"
    logger.info("Loading SAM model: %s ...", model_path)
    t = time.monotonic()
    _sam_model = FastSAM(model_path)
    elapsed = round((time.monotonic() - t) * 1000)
    logger.info("SAM model loaded in %dms", elapsed)

    # Warmup: dummy inference to initialize CUDA kernels
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _sam_model(dummy, device="cuda", retina_masks=False, conf=0.4, iou=0.9, verbose=False)
    logger.info("SAM CUDA warmup complete")


# ---------------------------------------------------------------------------
# Public interface — only function ai_pipeline.py calls
# ---------------------------------------------------------------------------

async def run_sam(image_bytes: bytes) -> dict[str, Any]:
    """
    Segment image into object regions.

    Args:
        image_bytes: Raw JPEG/PNG bytes.

    Returns:
        {
            "masks_found":    int,
            "bounding_boxes": list[dict],   # [{"x","y","w","h"}, ...]
            "crops":          list[bytes],  # JPEG bytes per detected region
            "latency_ms":     float,
        }
    """
    _validate_image(image_bytes)
    if settings.mock_sam:
        return await _run_mock(image_bytes)
    return await _run_fastsam(image_bytes)


# ---------------------------------------------------------------------------
# Implementation: Mock
# ---------------------------------------------------------------------------

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
    # crops=[] in mock — blip_service falls back to full image when empty
    return {"masks_found": n, "bounding_boxes": boxes, "crops": [], "latency_ms": latency}


# ---------------------------------------------------------------------------
# Implementation: FastSAM (real)
# ---------------------------------------------------------------------------

# Max regions to pass to BLIP — more regions = more BLIP calls = more latency
_MAX_CROPS = 4

# Minimum crop area as fraction of total image — skip tiny noise regions
_MIN_AREA_FRACTION = 0.02


async def _run_fastsam(image_bytes: bytes) -> dict[str, Any]:
    """
    Run FastSAM-s segmentation on the image.
    Returns bounding boxes + JPEG crops for the most significant regions.
    """
    if _sam_model is None:
        raise RuntimeError("SAM model not loaded. Call load_sam_model() at startup.")

    t = time.monotonic()
    logger.info("SAM [real] | input=%d bytes", len(image_bytes))

    result = await asyncio.to_thread(_fastsam_infer, image_bytes)

    latency = round((time.monotonic() - t) * 1000, 2)
    logger.info("SAM [real] done | masks=%d | %.1fms", result["masks_found"], latency)
    result["latency_ms"] = latency
    return result


def _fastsam_infer(image_bytes: bytes) -> dict[str, Any]:
    """CPU-thread: run FastSAM inference and extract top crops."""
    from PIL import Image

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Downscale oversized frames — reduces SAM + BLIP inference time significantly
    if max(pil_image.size) > settings.max_frame_size:
        pil_image.thumbnail((settings.max_frame_size, settings.max_frame_size))
    img_w, img_h = pil_image.size
    total_area = img_w * img_h

    results = _sam_model(
        pil_image,
        device="cuda",
        retina_masks=False,
        conf=0.4,
        iou=0.9,
        verbose=False,
    )

    boxes_raw = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            area_frac = (w * h) / total_area
            if area_frac >= _MIN_AREA_FRACTION:
                boxes_raw.append({"x": x1, "y": y1, "w": w, "h": h, "_area": w * h})

    # Sort by area descending, take top N
    boxes_raw.sort(key=lambda b: b["_area"], reverse=True)
    top_boxes = boxes_raw[:_MAX_CROPS]

    # Crop each region from original image
    crops = []
    clean_boxes = []
    for b in top_boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # Clamp to image bounds
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        crop = pil_image.crop((x, y, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        crops.append(buf.getvalue())
        clean_boxes.append({"x": x, "y": y, "w": w, "h": h})

    # If no regions found, return empty (blip_service will use full image)
    return {
        "masks_found": len(clean_boxes),
        "bounding_boxes": clean_boxes,
        "crops": crops,
    }
