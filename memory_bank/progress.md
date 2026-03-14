# Project Progress

## Phase Tracker
| Phase | Description | Status |
|---|---|---|
| Phase 0 | Memory Bank + Project Scaffold | COMPLETE |
| Phase 1 | Mock Pipeline + WebSocket Boilerplate | COMPLETE |
| Phase 1.5 | Real TTS + Real Ollama + Backend Hardening (B1-B4) | COMPLETE |
| Phase 2 | FastSAM / MobileSAM Integration | NOT STARTED |
| Phase 3 | CLIP / ResNet Classifier Integration | NOT STARTED |
| Phase 4 | Docker GPU Deploy + Cloud Migration | NOT STARTED |

## Completed Work Log
- [2026-03-11] Project initialized. All boilerplate files created.
  - cline_docs/ Memory Bank established
  - main.py, ai_pipeline.py, requirements.txt written
  - Dockerfile, docker-compose.yml written
  - core/config.py, core/logger.py written

- [2026-03-12] Real services activated + backend hardened.
  - Edge TTS integrated (tr-TR-EmelNeural, +30% speed)
  - Ollama integrated (llama3.2:3b, local HTTP API)
  - Turkish prompt with proper Unicode characters (ş,ı,ğ,ü,ö,ç)
  - LLM output cleaner (_clean_output) strips bullets/preamble/English parens
  - B1: Ollama warmup on startup via lifespan (pre-loads model, first frame fast)
  - B2: Frame validation — magic byte check for JPEG/PNG before pipeline entry
  - B3: Fast fail — Ollama connect timeout 3s, clear error messages
  - B4: Detailed /health — Ollama reachability probe, loaded models, mock flags

## What Works Right Now
- FastAPI WebSocket server starts and accepts connections
- 4-stage pipeline: SAM(mock) -> Classifier(mock) -> Ollama(real) -> TTS(real)
- Ollama model pre-loaded on startup — first real frame is fast
- Invalid image bytes rejected immediately with clear error (before pipeline)
- Ollama connection failures fail fast (3s) with actionable error messages
- /health shows full status: Ollama connectivity, model list, per-service mock flags
- Server sends JSON metadata + binary MP3 (2 messages per frame)
- Architecture confirmed: 1fps periodic capture, Edge TTS for audio, always-online
- Docker Compose ready (GPU passthrough configured, pending host toolkit install)

## Current .env State
```
MOCK_SAM=true          <- Phase 2 pending
MOCK_CLASSIFIER=true   <- Phase 3 pending
MOCK_TTS=false         <- Edge TTS active, tr-TR-EmelNeural, +30%
MOCK_OLLAMA=false      <- llama3.2:3b active
```

## Future Plans (not yet scheduled)

### OCR / Mode Switching (post Phase 3)
Architectural plan for a multi-mode pipeline:
- Add `mode` field to WebSocket binary message (1-byte prefix: 0x00=scene, 0x01=ocr)
- `ai_pipeline.py` branches into `run_scene_pipeline()` vs `run_ocr_pipeline()`
- `services/ocr_service.py` — new adapter (EasyOCR recommended for Turkish)
- New config fields: `mock_ocr: bool`, `ocr_engine: str = "easyocr"`
- **Proactive OCR offer**: In scene mode, if Classifier detects medicine/bottle/label,
  LLM prompt asks it to offer OCR ("Üzerindeki yazıyı okumamı ister misiniz?").
  Android receives `ocr_available: true` in JSON metadata and prompts user.
  User confirmation triggers a second frame with mode=ocr.
- OCR mode bypasses SAM+Classifier+Ollama entirely — faster, focused.

## Blockers / Pending Decisions
- SAM model choice (FastSAM vs MobileSAM) — pending teammate feedback
- Android client — handled separately by user or teammate
- Cloud deploy — needed before presentations; RunPod/Vast.ai GPU rental planned
