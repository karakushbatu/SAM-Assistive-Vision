# Project Progress

## Current Phase
- Phase 0: Scaffold + Memory Bank - done
- Phase 1: Mock pipeline + WebSocket - done
- Phase 2: Real STT + SAM + caption + LLM + TTS - done
- Phase 3: Android-ready backend contract - done
- Phase 4: GPU local runtime + performance tuning - done
- Phase 5: OCR branch refactor and evaluation - in progress
- Phase 6: Cloud deployment - pending
- Phase 7: Android client handoff - pending

## Current Active Pipeline

```text
image or audio+image
  -> optional faster-whisper STT
  -> FastSAM-s
  -> scene mode: BLIP-large -> qwen2.5:7b -> Edge TTS
  -> ocr mode: PaddleOCR -> qwen2.5:7b summary -> Edge TTS
```

## Important Replacements vs pipeline.jpg
- Mobile/OpenCV camera side replaced by Android-ready WebSocket upload contract.
- Vision path is `FastSAM-s + BLIP-large`.
- OCR branch is no longer only a plan; PaddleOCR backend was integrated.
- Legacy OCR fallback still exists as `OCR_BACKEND=ollama_vision`.
- LLM default remains `qwen2.5:7b`.

## Verified Runtime State (2026-04-16)
- `.venv` on Python `3.12.10`
- CUDA active with `torch 2.11.0+cu128`
- GPU: `RTX 5060 Ti 16GB`
- Ollama models verified:
  - `qwen2.5:7b`
  - `qwen3:4b`
  - `glm-ocr:latest`
  - `gemma3:4b`
  - `llama3.2:3b`

## Latest Measured Results

### Scene mode direct benchmark
- `ilaç.jpg`: about `7250 ms`
- `ss1_test1.jpg`: about `1907 ms`
- `yulaf.jpg`: about `2062 ms`
- average: about `3739 ms`

### OCR mode direct benchmark with PaddleOCR
- `ilaç.jpg`: about `22625 ms`
- `ss1_test1.jpg`: about `6343 ms`
- `yulaf.jpg`: about `19047 ms`
- average: about `16005 ms`
- OCR text output was empty on all three sample images.

## Current Bottlenecks
1. Scene mode:
   - Main bottleneck remains Ollama generation.
2. OCR mode:
   - PaddleOCR on current Windows local runtime falls back to CPU and misses text on provided samples.
3. TTS:
   - Edge TTS adds network-backed tail latency.

## What Changed Most Recently
- Integrated `PaddleOCR` as the new OCR backend.
- Added `OCR_BACKEND`, `PADDLEOCR_LANG`, and `PADDLEOCR_VERSION` config knobs.
- Added OCR warmup/readiness integration in `main.py`.
- Added Paddle cache isolation inside workspace for safer local runs.
- Re-tested `qwen3:4b` locally against `qwen2.5:7b`.
- Rewrote README to reflect the real current state and current testing commands.
- Direct testing showed:
  - `qwen3:4b` is not a good replacement yet in the current Ollama flow.
  - PaddleOCR is integrated, but not yet a good default for this Windows demo stack.

## Current Recommendation
- Keep `qwen2.5:7b` as default scene and OCR-summary model.
- Keep PaddleOCR support in the codebase.
- For a working local OCR demo right now, use:
  - `OCR_BACKEND=ollama_vision`
  - `OCR_MODEL=glm-ocr:latest`

## Next Practical Steps
1. Decide whether PaddleOCR stays default or becomes opt-in until Linux/cloud validation.
2. Add one more OCR backend candidate if needed:
   - EasyOCR or a lighter detector+recognizer stack.
3. Validate STT end-to-end with a real WAV sample over WebSocket.
4. Move to cloud only after OCR branch behavior is locked.
