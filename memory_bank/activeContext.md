# Active Context

## Current Phase
**Pipeline optimization with real runtime validation**

## Current Active State - 2026-04-16
- Scene pipeline is the stable branch:
  - `FastSAM-s -> BLIP-large -> qwen2.5:7b -> Edge TTS`
- STT is active when audio is sent through the WebSocket envelope.
- OCR branch was migrated to PaddleOCR in code.
- However current Windows local validation shows PaddleOCR is not yet production-ready in this stack.

## What Was Just Done
- Integrated PaddleOCR into `services/ocr_service.py`.
- Added OCR backend selection:
  - `OCR_BACKEND=paddleocr`
  - `OCR_BACKEND=ollama_vision`
- Added Paddle-specific config:
  - `PADDLEOCR_LANG`
  - `PADDLEOCR_VERSION`
- Added OCR warmup in `main.py` and exposed OCR backend state in `/health`.
- Downloaded missing BLIP cache files locally so scene tests can run again.
- Re-benchmarked local LLM candidates:
  - `qwen2.5:7b`
  - `qwen3:4b`

## Current Findings

### Scene branch
- Works correctly on current sample set.
- Latest direct benchmark:
  - `ilaç.jpg`: about `7.25s`
  - `ss1_test1.jpg`: about `1.91s`
  - `yulaf.jpg`: about `2.06s`
  - average: about `3.74s`
- Main bottleneck is still Ollama, not SAM or BLIP.

### OCR branch
- PaddleOCR loads and runs.
- On current Windows runtime, PaddleOCR falls back to CPU.
- On the current sample set, it returned empty OCR text for:
  - `ilaç.jpg`
  - `ss1_test1.jpg`
  - `yulaf.jpg`
- Average OCR branch time rose to about `16.0s`.

### LLM layer
- `qwen3:4b` exists locally and was benchmarked.
- In the current Ollama flow it did not outperform `qwen2.5:7b` in a usable way.
- `qwen2.5:7b` remains the correct default for now.

## Practical Decision Right Now
- Keep the PaddleOCR integration in the codebase.
- Do not switch the demo recommendation away from `qwen2.5:7b`.
- If a working OCR demo is needed immediately on this machine, use `OCR_BACKEND=ollama_vision`.

## Open Technical Questions
1. Is PaddleOCR worth keeping as default on Windows, or should it remain opt-in until Linux/cloud validation?
2. Should a second OCR candidate be added for comparison?
3. Should qwen3 family support be revisited later with a chat-specific integration path?
