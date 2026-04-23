# -*- coding: utf-8 -*-
"""
Interactive terminal demo for the backend.

This script behaves like a lightweight client:
    - asks for an image
    - optionally records a voice query
    - sends the payload over WebSocket
    - prints JSON output
    - saves and optionally plays MP3 output
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import struct
import sys
import time
from pathlib import Path

import soundfile as sf
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import settings


OUTPUT_DIR = REPO_ROOT / "tests" / "output"
SAMPLES_DIR = REPO_ROOT / "tests" / "sample_images"


def default_server_url() -> str:
    port = settings.port
    return f"ws://127.0.0.1:{port}/ws/vision"


def list_sample_images() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(
        path for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def choose_image_interactive() -> Path:
    samples = list_sample_images()
    print("\nMevcut örnek görseller:")
    for index, path in enumerate(samples, start=1):
        print(f"  {index}. {path.name}")
    print("  0. Kendi görsel yolumu gireceğim")

    while True:
        raw_choice = input("\nGörsel seçimi: ").strip()
        if raw_choice == "0":
            custom_path = Path(input("Görsel dosya yolu: ").strip('" '))
            if custom_path.is_file():
                return custom_path
            print("Dosya bulunamadı.")
            continue
        if raw_choice.isdigit():
            selected = int(raw_choice)
            if 1 <= selected <= len(samples):
                return samples[selected - 1]
        print("Geçerli bir seçim yap.")


def choose_audio_mode_interactive() -> tuple[str, Path | None, int]:
    print("\nSoru girişi:")
    print("  1. Mikrofondan sor")
    print("  2. WAV dosyası seç")
    print("  3. Sessiz gönder (sadece sahne analizi)")

    while True:
        choice = input("Seçim: ").strip()
        if choice == "1":
            seconds_raw = input("Kayıt süresi saniye [3]: ").strip()
            seconds = int(seconds_raw) if seconds_raw else 3
            return "mic", None, max(1, seconds)
        if choice == "2":
            audio_path = Path(input("WAV dosya yolu: ").strip('" '))
            if audio_path.is_file():
                return "wav", audio_path, 0
            print("WAV dosyası bulunamadı.")
            continue
        if choice == "3":
            return "none", None, 0
        print("Geçerli bir seçim yap.")


def record_wav_bytes(seconds: int, sample_rate: int = 16000) -> bytes:
    import sounddevice as sd

    print(f"\nKayıt başlıyor ({seconds}s). Şimdi sorunuzu söyleyin.")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Kayıt tamamlandı.")

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def build_payload(image_bytes: bytes, audio_bytes: bytes | None) -> bytes:
    if not audio_bytes:
        return image_bytes
    return struct.pack("<I", len(audio_bytes)) + audio_bytes + image_bytes


async def receive_json_event(ws, expected_statuses: set[str]) -> dict:
    while True:
        message = await ws.recv()
        if not isinstance(message, str):
            continue
        event = json.loads(message)
        if event.get("status") == "ping":
            continue
        if event.get("status") in expected_statuses:
            return event


async def receive_audio_response(ws, metadata: dict) -> bytes:
    if not metadata.get("audio_streaming"):
        message = await ws.recv()
        return message if isinstance(message, bytes) else b""

    chunks: list[bytes] = []
    while True:
        message = await ws.recv()
        if isinstance(message, bytes):
            chunks.append(message)
            continue
        event = json.loads(message)
        if event.get("status") == "ping":
            continue
        if event.get("status") == "audio_end":
            return b"".join(chunks)


def save_audio_file(audio_bytes: bytes) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_DIR / f"demo_terminal_{int(time.time())}.mp3"
    file_path.write_bytes(audio_bytes)
    return file_path


def play_audio_if_possible(file_path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(file_path))


def print_result(metadata: dict, mp3_bytes: bytes, audio_file: Path | None) -> None:
    pipeline = metadata.get("pipeline", {})
    print("\n--- Sonuç ---")
    print(f"Açıklama       : {metadata.get('description')}")
    print(f"Kullanıcı sesi : {metadata.get('user_query')}")
    print(f"Mod            : {metadata.get('mode')}")
    print(f"Toplam süre    : {metadata.get('total_latency_ms')} ms")
    print(f"BLIP caption   : {pipeline.get('blip', {}).get('caption')}")
    print(f"OCR preview    : {pipeline.get('ocr', {}).get('text_preview')}")
    print(f"OCR teklifi    : {metadata.get('ocr_available')}")
    print(f"Ses boyutu     : {len(mp3_bytes) // 1024} KB")
    if audio_file:
        print(f"Ses dosyası    : {audio_file}")


async def run_once(server_url: str, image_path: Path, audio_mode: str, audio_path: Path | None, mic_seconds: int, auto_play: bool) -> None:
    image_bytes = image_path.read_bytes()
    audio_bytes = None
    if audio_mode == "wav" and audio_path is not None:
        audio_bytes = audio_path.read_bytes()
    elif audio_mode == "mic":
        audio_bytes = record_wav_bytes(mic_seconds)

    payload = build_payload(image_bytes, audio_bytes)
    print(f"\nBağlanılıyor: {server_url}")
    print(f"Görsel        : {image_path}")
    if audio_mode == "mic":
        print("Ses girişi    : Mikrofon")
    elif audio_mode == "wav" and audio_path is not None:
        print(f"Ses girişi    : {audio_path}")
    else:
        print("Ses girişi    : Yok")

    try:
        async with websockets.connect(server_url, max_size=10 * 1024 * 1024) as ws:
            started = time.monotonic()
            await ws.send(payload)
            metadata = await receive_json_event(ws, {"ok", "busy", "error"})
            mp3_bytes = await receive_audio_response(ws, metadata)
            elapsed = round((time.monotonic() - started) * 1000, 2)
    except OSError as exc:
        raise SystemExit(
            "Backend baglantisi kurulamadi. "
            f"Server adresi: {server_url}. "
            "Backend'i once ayaga kaldir veya dogru portu kullan. "
            "Ornek: .\\.venv\\Scripts\\python -m uvicorn main:app --host 0.0.0.0 --port "
            f"{settings.port}"
        ) from exc

    if metadata.get("status") == "error":
        print(f"\nHata: {metadata.get('message')}")
        return
    if metadata.get("status") == "busy":
        print("\nSunucu meşgul, frame atlandı.")
        return

    metadata["client_elapsed_ms"] = elapsed
    audio_file = save_audio_file(mp3_bytes) if mp3_bytes else None
    print_result(metadata, mp3_bytes, audio_file)
    print(f"İstemci süresi : {elapsed} ms")

    if auto_play and audio_file is not None:
        play_audio_if_possible(audio_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminal demo client")
    parser.add_argument("--server", default=default_server_url())
    parser.add_argument("--image", default=None)
    parser.add_argument("--audio", default=None, help="Optional WAV file path")
    parser.add_argument("--mic-seconds", type=int, default=0, help="Record from microphone for N seconds")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.no_prompt:
        if not args.image:
            raise SystemExit("--no-prompt modunda --image zorunlu.")
        image_path = Path(args.image)
        if not image_path.is_file():
            raise SystemExit(f"Görsel bulunamadı: {image_path}")
        if args.audio:
            audio_mode = "wav"
            audio_path = Path(args.audio)
            mic_seconds = 0
        elif args.mic_seconds > 0:
            audio_mode = "mic"
            audio_path = None
            mic_seconds = args.mic_seconds
        else:
            audio_mode = "none"
            audio_path = None
            mic_seconds = 0
        await run_once(args.server, image_path, audio_mode, audio_path, mic_seconds, not args.no_play)
        return

    print("SAM Assistive Vision - Terminal Demo")
    print("OCR denemesi için mikrofonda 'yazıları oku' demen yeterli.")
    while True:
        image_path = Path(args.image) if args.image else choose_image_interactive()
        if not image_path.is_file():
            print("Geçerli bir görsel seç.")
            continue
        audio_mode, audio_path, mic_seconds = choose_audio_mode_interactive()
        await run_once(args.server, image_path, audio_mode, audio_path, mic_seconds, not args.no_play)
        again = input("\nYeni deneme yapılsın mı? [e/H]: ").strip().lower()
        if again not in {"e", "evet", "y"}:
            break


if __name__ == "__main__":
    asyncio.run(main())
