"""
Shared runtime coordination primitives.
"""

import asyncio

from core.config import settings


# Keep GPU inference predictable across WebSocket connections.
# Trade-off: lower throughput, better tail latency and lower VRAM spike risk.
gpu_inference_lock = asyncio.Semaphore(1)


def resolve_model_device() -> str:
    """Resolve model device from config and runtime torch availability."""
    requested = settings.model_device.lower()
    if requested in {"cpu", "cuda"}:
        return requested

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def torch_dtype_for_device(device: str):
    """Return a practical dtype for HuggingFace vision models."""
    import torch
    return torch.float16 if device == "cuda" else torch.float32
