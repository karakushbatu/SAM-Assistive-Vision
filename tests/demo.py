# -*- coding: utf-8 -*-
"""
SAM Assistive Vision — Demo Script
====================================
Demonstrates the full pipeline with real photos.

Usage:
    python tests/demo.py                        # uses sample_images/ or generates test scenes
    python tests/demo.py path/to/photo.jpg      # single photo
    python tests/demo.py --play                 # auto-play audio after each result (Windows)

Drop your own .jpg files into tests/sample_images/ for a real demo.
Audio output saved to tests/output/
"""

import asyncio
import io
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Terminal colors (works on Windows with modern terminal)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def header(text):   print(f"\n{BOLD}{CYAN}{text}{RESET}")
def ok(text):       print(f"  {GREEN}►{RESET} {text}")
def info(text):     print(f"  {DIM}{text}{RESET}")
def warn(text):     print(f"  {YELLOW}⚠ {text}{RESET}")
def divider():      print(f"  {DIM}{'─' * 58}{RESET}")


# ---------------------------------------------------------------------------
# Demo scenarios with voice query simulation
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name":        "Sahne Tanımlama",
        "description": "Kullanıcı kameraya nesneyi tutuyor.",
        "user_query":  None,
    },
]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images_from_dir(path: str) -> list[tuple[str, bytes]]:
    images = []
    for fname in sorted(os.listdir(path)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            with open(os.path.join(path, fname), "rb") as f:
                images.append((fname, f.read()))
    return images


def generate_demo_images() -> list[tuple[str, bytes]]:
    """Generate a few representable demo images when no real photos exist."""
    from PIL import Image, ImageDraw
    results = []

    # Scene 1: Office desk
    img = Image.new("RGB", (640, 480), (230, 220, 210))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 380, 640, 480], fill=(150, 130, 110))
    d.rectangle([80, 280, 500, 310], fill=(120, 80, 40))
    d.rectangle([80, 310, 110, 380], fill=(100, 65, 30))
    d.rectangle([470, 310, 500, 380], fill=(100, 65, 30))
    d.rectangle([180, 230, 350, 285], fill=(40, 40, 40))
    d.rectangle([185, 235, 345, 280], fill=(30, 100, 180))
    d.rectangle([400, 50, 580, 220], fill=(180, 220, 255))
    d.ellipse([360, 260, 390, 290], fill=(200, 150, 100))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
    results.append(("masa_laptop_kupa.jpg", buf.getvalue()))

    # Scene 2: Hallway
    img = Image.new("RGB", (640, 480), (200, 195, 185))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 380, 640, 480], fill=(140, 130, 110))
    d.rectangle([55, 95, 215, 385], fill=(180, 160, 140))
    d.rectangle([65, 105, 205, 375], fill=(100, 65, 40))
    d.ellipse([185, 230, 205, 250], fill=(200, 170, 50))
    d.rectangle([420, 100, 580, 280], fill=(180, 215, 255))
    d.line([500, 100, 500, 280], fill=(160, 160, 160), width=4)
    d.line([420, 190, 580, 190], fill=(160, 160, 160), width=4)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
    results.append(("koridor_kapi_pencere.jpg", buf.getvalue()))

    return results


# ---------------------------------------------------------------------------
# Core demo runner
# ---------------------------------------------------------------------------

async def run_demo(image_path: str | None, play_audio: bool):
    from services.blip_service import load_blip_model
    from services.stt_service import load_stt_model
    from services.sam_service import load_sam_model
    from services.ai_pipeline import run_full_pipeline
    from core.config import settings

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Startup ──────────────────────────────────────────────────────
    header("SAM Assistive Vision — Demo")
    print(f"  Pipeline: STT={not settings.mock_stt} | "
          f"SAM={'mock' if settings.mock_sam else 'REAL'} | "
          f"BLIP={not settings.mock_blip} | "
          f"Ollama={not settings.mock_ollama} | "
          f"TTS={not settings.mock_tts}")
    print(f"  Model: {settings.ollama_model} (num_ctx={settings.ollama_num_ctx}) | STT: {settings.stt_model} | Voice: {settings.tts_voice} {settings.tts_rate}")

    print("\n  Modeller yukleniyor...")
    t0 = time.monotonic()
    if not settings.mock_sam:
        await asyncio.to_thread(load_sam_model)
    if not settings.mock_blip:
        await asyncio.to_thread(load_blip_model)
    if not settings.mock_stt:
        await asyncio.to_thread(load_stt_model)
    info(f"Hazir ({round((time.monotonic()-t0)*1000)}ms)")

    # ── Load images ───────────────────────────────────────────────────
    if image_path:
        with open(image_path, "rb") as f:
            images = [(os.path.basename(image_path), f.read())]
    else:
        samples_dir = os.path.join(os.path.dirname(__file__), "sample_images")
        images = load_images_from_dir(samples_dir)
        if not images:
            warn("tests/sample_images/ bos — simule goruntular kullaniliyor")
            warn("Gercek demo icin JPG dosyalarinizi tests/sample_images/ klasorune koyun")
            images = generate_demo_images()

    # ── Run pipeline for each image ───────────────────────────────────
    for img_name, jpeg_bytes in images:
        history = []  # reset per image — no context bleed between different scenes
        header(f"Goruntu: {img_name}  ({len(jpeg_bytes)//1024}KB)")

        for i, scenario in enumerate(SCENARIOS):
            divider()
            ok(f"Senaryo {i+1}: {scenario['name']}")
            info(scenario["description"])

            # Inject user_query via STT mock override for demo purposes
            # (real STT requires actual audio bytes, so we simulate here)
            if scenario["user_query"]:
                info(f"Kullanici sorusu: \"{scenario['user_query']}\"")

            t = time.monotonic()
            result = await run_full_pipeline(
                jpeg_bytes,
                audio_bytes=None,   # no real audio for demo; query injected via history
                mode="scene",
                history=history if i > 0 else [],
            )
            total_ms = round((time.monotonic() - t) * 1000)

            m = result.metadata
            caption = m["pipeline"]["blip"]["caption"]
            description = m["description"]
            ocr = m.get("ocr_available", False)

            print(f"\n  {BOLD}BLIP caption:{RESET}  {caption}")
            print(f"  {BOLD}Turkce cevap:{RESET}  {description}")
            if ocr:
                print(f"  {YELLOW}{BOLD}OCR teklifi:{RESET}  Bu nesnede okunabilir yazi var!")

            blip_ms  = m["pipeline"]["blip"]["latency_ms"]
            o_ms     = m["pipeline"]["ollama"]["latency_ms"]
            tts_ms   = m["pipeline"]["tts"]["latency_ms"]
            audio_kb = m["pipeline"]["tts"]["audio_size_bytes"] // 1024

            info(f"Sureler: BLIP={blip_ms:.0f}ms | Ollama={o_ms:.0f}ms | TTS={tts_ms:.0f}ms | Toplam={total_ms}ms")
            info(f"Ses: {audio_kb}KB MP3")

            # Save audio
            audio_name = f"{img_name.rsplit('.', 1)[0]}_senaryo{i+1}.mp3"
            audio_path = os.path.join(output_dir, audio_name)
            with open(audio_path, "wb") as f:
                f.write(result.audio_bytes)
            info(f"Kaydedildi: tests/output/{audio_name}")

            if play_audio and sys.platform == "win32":
                os.startfile(audio_path)
                await asyncio.sleep(3.5)

            # Update history for context test
            history.append({"user": scenario["user_query"], "assistant": description})
            if len(history) > 3:
                history = history[-3:]

    # ── Summary ───────────────────────────────────────────────────────
    header("Demo Tamamlandi")
    print(f"  Ses dosyalari: tests/output/")
    print(f"  Gercek demo icin: tests/sample_images/ klasorune kendi fotograflarinizi ekleyin")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM Assistive Vision Demo")
    parser.add_argument("image", nargs="?", help="Tek bir JPEG dosyasi (opsiyonel)")
    parser.add_argument("--play", action="store_true", help="Sesi otomatik oynat (Windows)")
    args = parser.parse_args()

    asyncio.run(run_demo(args.image, args.play))
