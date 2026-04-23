# System Patterns

## 1. WebSocket Contract

### Client -> Server
- Endpoint: `WS /ws/vision`
- Supported binary payloads:
  - legacy image: raw `JPEG/PNG`
  - audio envelope:
    - `[4-byte little-endian audio_length][WAV audio bytes][JPEG/PNG image bytes]`

### Server -> Client
- First message:
  - JSON text event with `status=ok|busy|error`
- Second message:
  - single MP3 binary if `audio_streaming=false`
  - multiple MP3 chunks plus final `{"status":"audio_end"}` if `audio_streaming=true`

## 2. Pipeline Routing Pattern

### Scene mode
```text
image
  -> FastSAM-s
  -> BLIP-large
  -> qwen2.5:7b
  -> Edge TTS
```

### OCR mode
```text
image
  -> FastSAM-s
  -> glm-ocr:latest
  -> qwen2.5:7b summary
  -> Edge TTS
```

### Audio-aware path
```text
audio+image
  -> faster-whisper STT
  -> intent routing
  -> scene or ocr branch
```

## 3. Intent Routing Pattern
- `detect_intent(stt_text)` runs before expensive vision work when audio is present.
- Fast-path commands now require stricter matching to avoid misrouting free-form Turkish speech.
- Supported intents:
  - `scene`
  - `ocr`
  - `replay`
  - `mute`
  - `camera`

## 4. OCR Backend Pattern
- OCR is selectable at config level:
  - `OCR_BACKEND=ollama_vision`
  - `OCR_BACKEND=paddleocr`
- Reason:
  - allows side-by-side validation without changing the mobile contract
  - keeps OCR experiments isolated from scene-mode stability
- Current active recommendation:
  - `OCR_BACKEND=ollama_vision`
  - `OCR_MODEL=glm-ocr:latest`

## 5. OCR Recommendation Pattern
- OCR recommendation remains heuristic, not LLM-based.
- Source: `should_offer_ocr(blip_caption)`
- Reason:
  - avoids an extra Ollama call
  - keeps scene latency lower

## 6. GPU Pattern
- SAM and BLIP share a common `gpu_inference_lock`.
- Current Windows validation indicates PaddleOCR is not the active GPU-backed path in this setup.

## 7. Android Handoff Pattern
- Android side only needs:
  1. open WebSocket
  2. send image or audio+image frame
  3. read JSON result
  4. read MP3 or MP3 chunks
  5. optionally inspect `/contract`
- Current local Android-like clients:
  - `tests/ws_client_demo.py`
  - `android-test-shell/`

## 7b. Mobile Interaction Pattern
- Final intended app interaction is `Model A`.
- Pattern:
  1. camera preview stays open
  2. user presses a trigger button
  3. app captures the latest frame
  4. app records a short voice query
  5. both are sent together through the existing `audio+image` envelope
- Reason:
  - lower bandwidth than continuous streaming
  - simpler sync model
  - better fit for low-latency assistive feedback

## 8. Model Selection Pattern
- Scene mode and OCR-summary mode can use different LLMs:
  - `SCENE_OLLAMA_MODEL`
  - `OCR_SUMMARY_MODEL`
- OCR extraction backend is separate:
  - `OCR_BACKEND`
  - `OCR_MODEL`

## 9. Current Recommendation Pattern
- Stable demo branch:
  - scene mode with `qwen2.5:7b`
- Stable OCR branch:
  - `glm-ocr:latest` with `OCR_BACKEND=ollama_vision`
- Experimental branch:
  - PaddleOCR on current Windows local stack
- Voice quality recommendation:
  - `Edge TTS` for dev/demo fallback
  - premium TTS provider for final human-like voice quality
