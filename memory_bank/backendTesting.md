# Backend Testing Guide

## Goal
Validate services independently first, then validate the same WebSocket flow that Android will use.

## Fast Checks

```bash
python -m compileall main.py core services tests scripts
python tests/test_protocol.py
```

Expected:
- source files compile
- payload parser works
- Turkish intent routing works

## Server Startup

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expected:
- startup warms active models
- `/health` returns `status=healthy`
- `ready=true` only when required models are actually available

## Contract Check

```bash
curl http://localhost:8000/contract
```

## Per-Service Checks

### STT
Audio+image envelope gonder veya:

```bash
python tests/mic_test.py --seconds 5
```

Interaktif test:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_demo_terminal.ps1
```

Bu akista kullanici:
- bir gorsel secer
- isterse mikrofondan soru sorar
- isterse WAV verir
- backend sonucu terminalde gorur
- uretilen MP3 dosyasi `tests/output/` altina yazilir

Tek komutluk tam demo:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_full_demo.ps1
```

Not:
- Edge TTS baglantisi engellenirse frame artik hata vermez
- backend text sonucunu korur ve sessiz fallback MP3 dondurur

### Scene branch
```bash
python tests/test_pipeline_live.py --mode scene
```

Latest validated result on 2026-04-16:
- `ilaç.jpg`: about `7250 ms`
- `ss1_test1.jpg`: about `1907 ms`
- `yulaf.jpg`: about `2062 ms`
- average: about `3739 ms`

### OCR branch
```bash
python tests/test_pipeline_live.py --mode ocr
```

Current practical local choice:
- `OCR_BACKEND=ollama_vision`
- `OCR_MODEL=glm-ocr:latest`

Interpretation:
- PaddleOCR integration exists but is not the active demo path
- current Windows local demo should use Ollama vision OCR

### WebSocket / Android simulation

Image only:
```bash
python tests/ws_client_demo.py --host localhost --port 8000 --image tests/sample_images/yulaf.jpg --frames 1
```

Audio + image:
```bash
python tests/ws_client_demo.py --host localhost --port 8000 --image tests/sample_images/yulaf.jpg --audio path/to/query.wav --frames 1
```

### Android emulator shell

Android Studio ile:

1. `android-test-shell/` klasorunu ac
2. bir emulator baslat
3. backend'i lokalden ayaga kaldir
4. uygulamada `ws://10.0.2.2:8000/ws/vision` adresine baglan
5. gorsel ve opsiyonel WAV secerek test et

## Current Practical Recommendation

For a stable local demo:
- use scene mode
- keep `qwen2.5:7b`

For OCR demo on this exact Windows machine:
- use `OCR_BACKEND=ollama_vision`
- keep PaddleOCR only for later experiments
