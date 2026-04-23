# Project Progress

## Current Phase
- Phase 0: Scaffold + Memory Bank - done
- Phase 1: Mock pipeline + WebSocket - done
- Phase 2: Real STT + SAM + caption + LLM + TTS - done
- Phase 3: Android-ready backend contract - done
- Phase 4: GPU local runtime + performance tuning - done
- Phase 5: Demo hardening and Android test shell - in progress
- Phase 6: Cloud deployment - pending

## Current Active Pipeline

```text
image or audio+image
  -> optional faster-whisper STT
  -> FastSAM-s
  -> scene mode: BLIP-large -> qwen2.5:7b -> Edge TTS
  -> ocr mode: glm-ocr:latest -> qwen2.5:7b summary -> Edge TTS
```

## Verified Runtime State
- `.venv` on Python `3.12.10`
- CUDA active with `torch 2.11.0+cu128`
- GPU: `RTX 5060 Ti 16GB`
- Ollama models verified:
  - `qwen2.5:7b`
  - `qwen3:4b`
  - `glm-ocr:latest`

## Latest Working Results

### Scene mode direct benchmark
- `ilaç.jpg`: about `7250 ms`
- `ss1_test1.jpg`: about `1907 ms`
- `yulaf.jpg`: about `2062 ms`
- average: about `3739 ms`

### OCR practical decision
- `PaddleOCR` code path was integrated and evaluated.
- Current Windows local runtime produced:
  - CPU fallback
  - empty OCR output
  - higher latency
- Active working recommendation returned to:
  - `OCR_BACKEND=ollama_vision`
  - `OCR_MODEL=glm-ocr:latest`

## What Changed Most Recently
- Switched default OCR backend back to `ollama_vision`.
- Removed PaddleOCR from active install requirements.
- Added `.cache/` and Android shell build outputs to `.gitignore`.
- Added interactive terminal demo:
  - `scripts/demo_terminal.py`
  - `scripts/run_demo_terminal.ps1`
- Added one-command full demo wrapper:
  - `scripts/run_full_demo.ps1`
- Added Android emulator test shell scaffold:
  - `android-test-shell/`
- Added sample audio folder guidance:
  - `tests/sample_audio/README.md`
- Removed unused legacy files:
  - `services/classifier_service.py`
  - `tests/demo.py`
- Hardened TTS failure handling:
  - if Edge TTS is blocked or unavailable, the pipeline now returns text result and a silent fallback MP3 instead of failing the whole frame
- Fixed an STT/intent routing false positive:
  - free-form Turkish speech was sometimes misrouted into fast-path commands
  - command matching is now stricter
- Hardened full demo startup:
  - `scripts/run_full_demo.ps1` now fails if the backend process exits before readiness instead of silently attaching to an old server on the same port
- Fixed Android test shell Gradle setup for AGP 9:
  - removed `org.jetbrains.kotlin.android` plugin
  - removed legacy `android.kotlinOptions {}` usage
  - aligned Compose plugin version with AGP 9 built-in Kotlin baseline
- Updated Android shell Gradle wrapper:
  - `gradle-wrapper.properties` now uses Gradle `9.3.1`
- Enabled cleartext traffic for the Android test shell:
  - local emulator can now connect to `ws://10.0.2.2:<port>/ws/vision`
- Final mobile interaction model was chosen:
  - `Model A`
  - user presses a trigger button
  - app captures the latest frame and a short audio query in the same interaction
- TTS direction was clarified:
  - `Edge TTS` remains the local/dev fallback
  - a more human production voice should come from a premium provider later
- Local Edge TTS profile was updated:
  - `tr-TR-AhmetNeural`
  - `TTS_RATE=+10%`
- Terminal demo client now reads the default WebSocket port from `.env` / settings
- Terminal demo connection failures now return a clearer startup hint instead of a raw traceback

## Current Recommendation
- Keep `qwen2.5:7b` as the default LLM.
- Keep scene mode as the main demo path.
- Use `ollama_vision` for OCR demos until a better local OCR backend is validated.
- Use `Model A` for the final Android interaction design.
- Treat `Edge TTS` as dev/demo quality, not final product voice quality.

## Next Practical Steps
1. Validate STT over WebSocket with microphone or sample WAV.
2. Open `android-test-shell` in Android Studio and run on an emulator.
3. Decide whether WAV-based Android testing is enough for the next demo, or whether live microphone capture is required.
4. Only after that, revisit cloud deployment.
