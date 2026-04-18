"""
Protocol unit smoke tests.

Run:
    python tests/test_protocol.py
"""

import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import _parse_frame_payload
from services.intent_service import detect_intent


JPEG = b"\xff\xd8\xff" + b"image"
PNG = b"\x89PNG" + b"image"


def test_raw_jpeg():
    image, audio, protocol = _parse_frame_payload(JPEG)
    assert image == JPEG
    assert audio is None
    assert protocol == "legacy_image"


def test_raw_png():
    image, audio, protocol = _parse_frame_payload(PNG)
    assert image == PNG
    assert audio is None
    assert protocol == "legacy_image"


def test_audio_envelope():
    audio_bytes = b"a" * 8000
    payload = struct.pack("<I", len(audio_bytes)) + audio_bytes + JPEG
    image, audio, protocol = _parse_frame_payload(payload)
    assert image == JPEG
    assert audio == audio_bytes
    assert protocol == "audio_image_envelope"


def test_short_audio_is_ignored():
    audio_bytes = b"a" * 100
    payload = struct.pack("<I", len(audio_bytes)) + audio_bytes + JPEG
    image, audio, protocol = _parse_frame_payload(payload)
    assert image == JPEG
    assert audio is None
    assert protocol == "audio_image_envelope"


def test_invalid_envelope_fails():
    payload = struct.pack("<I", 9999) + b"bad"
    try:
        _parse_frame_payload(payload)
    except ValueError:
        return
    raise AssertionError("Invalid envelope should fail")


def test_turkish_intents():
    assert detect_intent("Kamerayı aç")["intent"] == "camera"
    assert detect_intent("Tekrar söyle")["intent"] == "replay"
    assert detect_intent("Yazıları oku")["intent"] == "ocr"
    assert detect_intent("Ne görüyorsun?")["intent"] == "scene"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("protocol tests ok")
