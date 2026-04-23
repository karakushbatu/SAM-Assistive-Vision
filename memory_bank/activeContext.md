# Active Context

## Current Phase
**Stabilization, demo readiness, and Android-like testing**

## Current Active State - 2026-04-23
- Stable scene branch:
  - `FastSAM-s -> BLIP-large -> qwen2.5:7b -> Edge TTS`
- Active OCR branch for real local usage:
  - `FastSAM-s -> glm-ocr:latest -> qwen2.5:7b -> Edge TTS`
- STT is active when the client sends the WebSocket audio+image envelope.
- Primary local validation target is now a working demo flow, not further OCR backend experiments.

## What Was Just Done
- Reverted default OCR recommendation back to `ollama_vision`.
- Removed PaddleOCR from active install requirements.
- Added interactive terminal demo client:
  - `scripts/demo_terminal.py`
  - `scripts/run_demo_terminal.ps1`
- Added one-command full demo wrapper:
  - `scripts/run_full_demo.ps1`
- Added minimal Android emulator test shell:
  - `android-test-shell/`
- Android shell Gradle setup was aligned with AGP 9 plugin rules.
- Legacy `kotlinOptions` usage was removed from the Android shell build file.
- Android shell wrapper was updated to Gradle `9.3.1` to satisfy AGP requirements.
- Android shell manifest now allows local cleartext WebSocket traffic for emulator testing.
- Added sample audio guidance:
  - `tests/sample_audio/README.md`
- Removed clearly unused legacy files:
  - `services/classifier_service.py`
  - `tests/demo.py`

## Current Findings

### Scene branch
- Stable and demo-ready on the current machine.
- Main latency source is still Ollama generation.

### OCR branch
- PaddleOCR remains only an experimental code path.
- Current local demo/default path is again `glm-ocr:latest` through Ollama.
- This is the practical choice until a better OCR backend is validated in a future environment.

### Demo flow
- Terminal-driven backend demo is now available.
- A one-command full demo flow is available and starts/stops the backend automatically.
- Android-style integration can be simulated in two ways:
  - `tests/ws_client_demo.py`
  - `android-test-shell/` on an emulator
- Fast-path voice command routing was tightened after a false positive during WAV testing.

### TTS robustness
- Edge TTS is still the primary TTS path.
- If Edge TTS is blocked by network or policy, the backend now keeps the text result and returns a silent fallback MP3 instead of failing the frame.

## Practical Decision Right Now
- Keep `qwen2.5:7b` as default.
- Keep `OCR_BACKEND=ollama_vision` as default.
- Use the terminal demo for immediate mentor/demo runs.
- Use the Android test shell for early Android integration checks.
- Final mobile UX direction is `Model A`:
  - capture the latest frame
  - capture a short audio query
  - send both together in one request
- `Edge TTS` stays as the working fallback, now with a more natural local profile:
  - `tr-TR-AhmetNeural`
  - `TTS_RATE=+10%`
- Final product voice quality should still move to a stronger TTS provider later.
- Terminal demo startup is now aligned with configured backend port and gives a clearer connection error message.

## Open Technical Questions
1. Do we want live microphone capture inside the Android shell, or is WAV-based testing enough for the next milestone?
2. Should OCR remain demo-only for now, or do we need a stronger OCR backend before cloud work starts?
3. When Android handoff starts, do we want to freeze the WebSocket contract as v1 and avoid further shape changes?
