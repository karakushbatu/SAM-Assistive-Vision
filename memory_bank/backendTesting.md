# Backend Testing Guide

## Goal
Validate services independently first, then validate the same WebSocket flow that Android will use.

## Fast Checks

```bash
python -m compileall main.py core services tests
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

Latest validated result on 2026-04-16 with PaddleOCR:
- `ilaç.jpg`: about `22625 ms`
- `ss1_test1.jpg`: about `6343 ms`
- `yulaf.jpg`: about `19047 ms`
- OCR text was empty on all tested sample images

Interpretation:
- PaddleOCR integration exists
- current Windows local runtime is not a good OCR demo baseline yet

### WebSocket / Android simulation

Image only:
```bash
python tests/ws_client_demo.py --host localhost --port 8000 --image tests/sample_images/yulaf.jpg --frames 1
```

Audio + image:
```bash
python tests/ws_client_demo.py --host localhost --port 8000 --image tests/sample_images/yulaf.jpg --audio path/to/query.wav --frames 1
```

## Current Practical Recommendation

For a stable local demo:
- use scene mode
- keep `qwen2.5:7b`

For OCR demo on this exact Windows machine:
- try PaddleOCR if you want to inspect the new integration
- use `OCR_BACKEND=ollama_vision` if you need a more reliable OCR result today
