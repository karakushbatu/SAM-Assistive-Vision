# Roadmap

## Current State
- Backend runs locally with real STT, real FastSAM-s, real BLIP-large, real Ollama, and real Edge TTS.
- CUDA is active in `.venv`.
- Android-ready contract endpoint exists: `GET /contract`.
- Main bottleneck is now Ollama latency.

## Immediate Next Steps
1. **Ollama latency reduction**
   - Benchmark lower-cost local variants such as `qwen2.5:3b` only if Turkish quality improves enough.
   - Try shorter prompt variants for OCR check and description prompt.
   - Evaluate reducing the OCR-check call frequency when the scene is clearly non-text-heavy.

2. **Android integration handoff**
   - Share `/contract` JSON and `tests/ws_client_demo.py` behavior with Android developer.
   - Define the first mobile client milestone: image-only WebSocket flow.
   - Define the second milestone: audio+image envelope flow.

3. **OCR branch**
   - Implement real OCR mode behind current `ocr` intent.
   - Use EasyOCR or another practical OCR engine on SAM crop output.
   - Return concise Turkish summaries for medicine boxes, labels, menus, and packaged products.

4. **Cloud-readiness**
   - Align Docker runtime with validated local CUDA stack.
   - Choose GPU target platform: RunPod, Vast.ai, or another GPU VM.
   - Add deployment-focused env template for non-local networking and public base URL handling.

## Recommended Execution Order
1. Freeze current working local baseline.
2. Add OCR branch with minimal API changes.
3. Hand off contract and sample client behavior to Android side.
4. Run one integrated Android-emulator smoke test.
5. Move the validated stack to a GPU cloud instance.

## Demo Scope Recommendation
- Keep the demo narrow and reliable:
  1. show `/health`
  2. show `/contract`
  3. send one sample image through WebSocket
  4. play Turkish MP3 response
  5. optionally show one OCR-triggering medicine image after OCR branch lands

## Risks To Watch
- Ollama latency may dominate the perceived responsiveness even after GPU optimization.
- Windows console encoding can mislead debugging; validate Unicode with JSON or file output when needed.
- Docker and cloud runtimes must not silently fall back to CPU wheels.
