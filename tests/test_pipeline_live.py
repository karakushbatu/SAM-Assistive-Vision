# -*- coding: utf-8 -*-
"""
Live pipeline test — sends real JPEG images directly through the pipeline
(no WebSocket needed, calls run_full_pipeline directly).

Usage:
    python tests/test_pipeline_live.py

Creates test images with PIL and optionally loads real photos from
tests/sample_images/ if they exist.

Output: prints descriptions + saves MP3 files to tests/output/
"""

import asyncio
import io
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Test image generators
# ---------------------------------------------------------------------------

def make_office_scene() -> bytes:
    """Simulate an office scene: desk, chair, laptop, window."""
    img = Image.new("RGB", (640, 480), color=(230, 220, 210))
    d = ImageDraw.Draw(img)
    # Floor
    d.rectangle([0, 380, 640, 480], fill=(160, 140, 120))
    # Desk
    d.rectangle([80, 280, 500, 310], fill=(120, 80, 40))
    d.rectangle([80, 310, 110, 380], fill=(100, 65, 30))
    d.rectangle([470, 310, 500, 380], fill=(100, 65, 30))
    # Laptop on desk
    d.rectangle([180, 230, 350, 285], fill=(40, 40, 40))
    d.rectangle([185, 235, 345, 280], fill=(30, 100, 180))
    # Chair
    d.rectangle([520, 220, 620, 300], fill=(80, 60, 40))
    d.rectangle([530, 300, 610, 380], fill=(80, 60, 40))
    # Window on wall
    d.rectangle([400, 50, 580, 220], fill=(180, 220, 255))
    d.line([490, 50, 490, 220], fill=(150, 150, 150), width=3)
    d.line([400, 135, 580, 135], fill=(150, 150, 150), width=3)
    # Coffee cup
    d.ellipse([360, 260, 390, 290], fill=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_hallway() -> bytes:
    """Simulate a hallway with door and window."""
    img = Image.new("RGB", (640, 480), color=(200, 195, 185))
    d = ImageDraw.Draw(img)
    # Floor
    d.rectangle([0, 380, 640, 480], fill=(140, 130, 110))
    # Door (left)
    d.rectangle([60, 100, 210, 380], fill=(120, 80, 50))
    d.rectangle([65, 105, 205, 375], fill=(100, 65, 40))
    d.ellipse([185, 230, 205, 250], fill=(200, 170, 50))  # door handle
    # Door frame
    d.rectangle([55, 95, 215, 385], fill=(180, 160, 140), width=8)
    # Window (right)
    d.rectangle([420, 100, 580, 280], fill=(180, 215, 255))
    d.line([500, 100, 500, 280], fill=(160, 160, 160), width=4)
    d.line([420, 190, 580, 190], fill=(160, 160, 160), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_street_scene() -> bytes:
    """Simulate outdoor: pavement, parked car, bicycle."""
    img = Image.new("RGB", (640, 480), color=(180, 200, 160))
    d = ImageDraw.Draw(img)
    # Sky
    d.rectangle([0, 0, 640, 200], fill=(135, 185, 230))
    # Road
    d.rectangle([0, 350, 640, 480], fill=(100, 100, 100))
    d.line([0, 415, 640, 415], fill=(255, 255, 255), width=4)
    # Car
    d.rectangle([50, 280, 300, 370], fill=(180, 60, 60))
    d.rectangle([80, 250, 270, 290], fill=(160, 50, 50))
    d.ellipse([80, 355, 130, 385], fill=(30, 30, 30))
    d.ellipse([210, 355, 260, 385], fill=(30, 30, 30))
    # Bicycle (right side)
    d.ellipse([460, 310, 520, 370], fill=(0, 0, 0), width=5)
    d.ellipse([540, 310, 600, 370], fill=(0, 0, 0), width=5)
    d.line([490, 340, 570, 340], fill=(80, 80, 80), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def load_sample_images() -> list[tuple[str, bytes]]:
    """Load from tests/sample_images/ if available, otherwise use generated ones."""
    samples_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    images = []

    if os.path.isdir(samples_dir):
        for fname in sorted(os.listdir(samples_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(samples_dir, fname)
                with open(path, "rb") as f:
                    images.append((fname, f.read()))
                print(f"  Loaded real image: {fname}")

    if not images:
        print("  No real images found, using generated test scenes.")
        images = [
            ("office_scene.jpg",  make_office_scene()),
            ("hallway.jpg",       make_hallway()),
            ("street_scene.jpg",  make_street_scene()),
        ]

    return images


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

async def run_tests(mode: str, skip_tts: bool):
    from services.blip_service import load_blip_model
    from services.sam_service import load_sam_model
    from services.stt_service import load_stt_model
    from services.ai_pipeline import run_full_pipeline

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== SAM Assistive Vision — Pipeline Test ===\n")

    # Load models (no-op if already loaded in this process)
    from core.config import settings
    if mode == "scene" and not settings.mock_blip:
        print("BLIP modeli yukleniyor (cache'den ~5s)...")
        await asyncio.to_thread(load_blip_model)
    if not settings.mock_sam:
        print("SAM modeli yukleniyor (cache'den ~2s)...")
        await asyncio.to_thread(load_sam_model)
    if not settings.mock_stt:
        print("STT modeli yukleniyor (cache'den ~4s)...")
        await asyncio.to_thread(load_stt_model)

    images = load_sample_images()
    print(f"\n{len(images)} goruntu test edilecek\n")
    print("-" * 60)

    results = []
    for name, jpeg_bytes in images:
        print(f"\n[{name}] ({len(jpeg_bytes)} bytes)")
        t = time.monotonic()

        result = await run_full_pipeline(
            jpeg_bytes,
            audio_bytes=None,
            mode=mode,
            skip_tts=skip_tts,
        )
        total = round((time.monotonic() - t) * 1000)

        m = result.metadata
        caption = m["pipeline"]["blip"]["caption"]
        ocr_preview = m["pipeline"]["ocr"]["text_preview"]
        description = m["description"]
        blip_ms = m["pipeline"]["blip"]["latency_ms"]
        ocr_ms = m["pipeline"]["ocr"]["latency_ms"]
        ollama_ms = m["pipeline"]["ollama"]["latency_ms"]
        tts_ms = m["pipeline"]["tts"]["latency_ms"]
        audio_kb = m["pipeline"]["tts"]["audio_size_bytes"] // 1024

        if caption:
            print(f"  BLIP caption  : {caption}")
        if ocr_preview:
            print(f"  OCR preview   : {ocr_preview}")
        print(f"  Turkish desc  : {description}")
        print(f"  OCR available : {m.get('ocr_available', False)}")
        print(
            "  Latency       : "
            f"BLIP={blip_ms:.0f}ms | OCR={ocr_ms:.0f}ms | "
            f"Ollama={ollama_ms:.0f}ms | TTS={tts_ms:.0f}ms | Total={total}ms"
        )
        print(f"  Audio         : {audio_kb}KB MP3")

        # Save audio
        audio_path = os.path.join(output_dir, name.replace(".jpg", ".mp3").replace(".jpeg", ".mp3").replace(".png", ".mp3"))
        if result.audio_bytes:
            with open(audio_path, "wb") as f:
                f.write(result.audio_bytes)
            print(f"  Saved audio   : {audio_path}")
        else:
            print("  Saved audio   : yok (skip_tts=true)")

        results.append({"name": name, "caption": caption, "description": description, "total_ms": total})

    print("\n" + "=" * 60)
    print("OZET:")
    for r in results:
        print(f"  {r['name']:30} -> {r['description'][:60]}")
    avg_ms = sum(r["total_ms"] for r in results) // len(results)
    print(f"\nOrtalama pipeline suresi: {avg_ms}ms ({avg_ms/1000:.1f}s)")
    print("\nTest tamamlandi. Ses dosyalari: tests/output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live pipeline tests on sample images")
    parser.add_argument("--mode", choices=["scene", "ocr"], default="scene")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_tests(args.mode, args.skip_tts))
