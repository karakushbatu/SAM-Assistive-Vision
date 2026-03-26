# -*- coding: utf-8 -*-
"""
Microphone STT Test — Whisper Demo
====================================
Microfondan ses kaydeder ve faster-whisper ile Türkçe transkript çıkarır.
Pipeline'a entegre olmadan sadece STT katmanını test eder.

Usage:
    python tests/mic_test.py                  # 3 saniye kayıt
    python tests/mic_test.py --seconds 5      # 5 saniye kayıt
    python tests/mic_test.py --loop           # Sürekli kayıt (Ctrl+C ile dur)
"""

import asyncio
import io
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 16000  # faster-whisper expects 16kHz


def record_audio(seconds: int) -> bytes:
    """Microfondan WAV bytes kaydeder."""
    import sounddevice as sd
    import soundfile as sf

    print(f"  Kayıt başlıyor ({seconds}s)... Konuşun!")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  Kayıt bitti.")

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return buf.getvalue()


async def transcribe(wav_bytes: bytes) -> str | None:
    """WAV bytes'ı STT servisine gönderir."""
    from services.stt_service import load_stt_model, run_stt
    from core.config import settings

    if not settings.mock_stt:
        # Model zaten yüklüyse no-op, değilse yükle
        await asyncio.to_thread(load_stt_model)

    result = await run_stt(wav_bytes)
    return result["text"], result["latency_ms"]


async def run(seconds: int, loop_mode: bool):
    from core.config import settings

    print("\n=== Whisper STT Demo ===")
    print(f"  Model : faster-whisper/{settings.stt_model}")
    print(f"  Mod   : {'GERÇEK (CUDA)' if not settings.mock_stt else 'MOCK'}")
    print(f"  Süre  : {seconds}s / kayıt\n")

    # Load model once
    if not settings.mock_stt:
        print("  Model yükleniyor...")
        from services.stt_service import load_stt_model
        await asyncio.to_thread(load_stt_model)
        print("  Model hazır.\n")

    iteration = 0
    while True:
        iteration += 1
        if loop_mode:
            print(f"─── Kayıt #{iteration} ───────────────────────────")

        wav_bytes = record_audio(seconds)

        t = time.monotonic()
        text, latency_ms = await transcribe(wav_bytes)
        total_ms = round((time.monotonic() - t) * 1000)

        if text:
            print(f"  ✔ Transkript : \"{text}\"")
        else:
            print(f"  — Ses algılanmadı (sessizlik / konuşma yok)")
        print(f"  ⏱ STT süresi : {latency_ms:.0f}ms")
        print()

        if not loop_mode:
            break

        try:
            input("  [Enter] ile devam et, [Ctrl+C] ile çık... ")
        except KeyboardInterrupt:
            print("\n  Çıkılıyor.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Mic STT Test")
    parser.add_argument("--seconds", default=3, type=int, help="Kayıt süresi (varsayılan: 3s)")
    parser.add_argument("--loop",    action="store_true",   help="Tekrar tekrar kayıt yap")
    args = parser.parse_args()

    asyncio.run(run(args.seconds, args.loop))
