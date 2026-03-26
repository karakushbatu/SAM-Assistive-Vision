# -*- coding: utf-8 -*-
"""
WebSocket Client Demo — Android Simulator
==========================================
Simulates the Android client connecting to the backend over WebSocket.
Sends JPEG frames, receives JSON + MP3, plays audio.

Bu script Android uygulamasının yapacağı tam olarak budur:
  1. WebSocket bağlantısı aç
  2. JPEG frame gönder
  3. JSON metadata al (metin açıklama)
  4. MP3 binary al → seslendir

Usage:
    # Önce sunucuyu başlat:
    python -m uvicorn main:app --host 0.0.0.0 --port 8000

    # Sonra bu client'ı çalıştır:
    python tests/ws_client_demo.py
    python tests/ws_client_demo.py --image path/to/photo.jpg
    python tests/ws_client_demo.py --host 192.168.x.x  # uzak sunucu
    python tests/ws_client_demo.py --frames 5          # 5 frame gönder (1fps simüle)
"""

import asyncio
import io
import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_test_jpeg(width=640, height=480) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (210, 200, 190))
    d = ImageDraw.Draw(img)
    d.rectangle([0, height//2, width, height], fill=(140, 130, 110))
    d.rectangle([80, height//3, 400, height//2+20], fill=(120, 80, 40))
    d.rectangle([180, height//3-60, 320, height//3+5], fill=(40, 40, 40))
    d.rectangle([185, height//3-55, 315, height//3], fill=(30, 100, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def run_client(host: str, port: int, image_path: str | None, frame_count: int):
    import websockets

    url = f"ws://{host}:{port}/ws/vision"
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load JPEG
    if image_path and os.path.isfile(image_path):
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
        print(f"Goruntu: {image_path} ({len(jpeg_bytes)//1024}KB)")
    else:
        print("Test gorseli olusturuluyor...")
        jpeg_bytes = make_test_jpeg()

    print(f"Sunucu: {url}")
    print(f"Frame sayisi: {frame_count}")
    print("-" * 55)

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        print("WebSocket baglantisi acildi\n")

        for frame_num in range(1, frame_count + 1):
            print(f"[Frame {frame_num}/{frame_count}] JPEG gonderiliyor ({len(jpeg_bytes)} bytes)...")
            t_send = time.monotonic()

            await ws.send(jpeg_bytes)

            # Message 1: JSON metadata
            json_msg = await ws.recv()
            metadata = json.loads(json_msg)

            # Message 2: MP3 audio
            mp3_bytes = await ws.recv()

            total_ms = round((time.monotonic() - t_send) * 1000)

            if metadata.get("status") == "error":
                print(f"  HATA: {metadata.get('message')}")
                continue

            print(f"  BLIP caption  : {metadata['pipeline']['blip']['caption']}")
            print(f"  Turkce aciklama: {metadata['description']}")
            print(f"  OCR teklifi   : {metadata.get('ocr_available', False)}")
            print(f"  Toplam sure   : {total_ms}ms  (server-side: {metadata['total_latency_ms']:.0f}ms)")
            print(f"  Ses           : {len(mp3_bytes)//1024}KB MP3")

            # Save MP3
            mp3_path = os.path.join(output_dir, f"ws_frame_{frame_num}.mp3")
            with open(mp3_path, "wb") as f:
                f.write(mp3_bytes)
            print(f"  Kaydedildi    : tests/output/ws_frame_{frame_num}.mp3")

            if sys.platform == "win32":
                os.startfile(mp3_path)

            print()

            # Simulate 1fps — wait before next frame
            if frame_num < frame_count:
                elapsed = time.monotonic() - t_send
                wait = max(0, 1.0 - elapsed)
                if wait > 0:
                    print(f"  (1fps icin {wait:.1f}s bekleniyor...)\n")
                    await asyncio.sleep(wait)

    print("Baglanti kapatildi.")
    print(f"Tum MP3 dosyalar: tests/output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Android WebSocket Client Simulator")
    parser.add_argument("--host",   default="localhost",    help="Sunucu IP (varsayilan: localhost)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (varsayilan: 8000)")
    parser.add_argument("--image",  default=None,           help="JPEG dosya yolu")
    parser.add_argument("--frames", default=3,   type=int,  help="Gonderilecek frame sayisi")
    args = parser.parse_args()

    asyncio.run(run_client(args.host, args.port, args.image, args.frames))
