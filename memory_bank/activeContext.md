# Active Context

## Current Phase
**Pipeline Optimization — All Stages Real, Focus on Quality & Speed**

## What Was Just Done (2026-03-26) — Optimization Pass

- **Moondream2 attempted, reverted to BLIP-large:**
  - rev 2025-01-09: requires libvips native DLL (not available on Windows via pip)
  - rev 2024-08-26: PhiConfig incompatible with transformers 5.x
  - BLIP-large restored — proven working, bottleneck is Ollama not BLIP anyway
- Adaptive frame resize: `MAX_FRAME_SIZE=640` — images > 640px thumbnailed before SAM + BLIP ✅
- Streaming TTS: `TTS_STREAMING=true` — Edge TTS chunks sent over WebSocket as generated ✅
  - ai_pipeline.py: `skip_tts=True` parameter; tts_service.py: `stream_tts()` async generator
  - main.py: streams chunks, collects bytes for replay
- Frame skip lock: per-connection `asyncio.Lock()` — pipeline busy → `{"status":"busy"}` + silent MP3 ✅
- WebSocket heartbeat: `_ping_loop()` task, ping every 20s → prevents connection timeout ✅
- /health endpoint: added `gpu` (VRAM used/total/device), `models_loaded`, `active_connections` ✅
- Pipeline tested end-to-end with real photos — all results correct ✅
  - İlaç/Kreon: OCR teklifi tetiklendi, marka tanındı
  - FPS screenshot: OCR false, silah + saat kulesi doğru
  - Yulaf: tüm nesneler doğru Türkçe çevrildi

## What Was Done Earlier (2026-03-26)
- Phase 3: Real faster-whisper tiny STT activated (CUDA, ~35ms warm inference)
- tests/mic_test.py created — microphone recording demo via sounddevice
- intent_service.py created — Turkish keyword → scene/ocr/replay/mute/camera dispatch
- Phase 4: Real FastSAM-s integrated (ultralytics) — top 4 crops by area, min 2% size filter
- FastSAM warmup with CUDA dummy inference added to main.py lifespan
- BLIP multi-crop mode: each SAM crop captioned separately, top 3 joined with "; "
- "araf" BLIP hallucination filter added
- LLM switched from llama3.2:3b → qwen2.5:7b (better Turkish + brand/drug knowledge)
- Phase 5: Conversation history injected into Ollama prompt (last 2 previous descriptions)
- 2-call Ollama architecture: fast OCR yes/no check → then description generation
- Proactive OCR offer: "Üzerindeki yazıları okumamı ister misiniz?" appended post-processing
- Chinese character hallucination fixed: "SADECE Türkçe yaz. Asla Çince..." in system prompts
- .env: all MOCK flags = false, OLLAMA_MODEL=qwen2.5:7b

## Current Active State
- All pipeline stages REAL: STT + SAM + BLIP + Ollama + TTS
- Server startup: FastSAM warmup (~1s) + BLIP warmup (~5s cache) + Ollama warmup (~1s)
- Pipeline latency warm: ~3-4s/frame total
  - STT: ~35ms (faster-whisper tiny, CUDA)
  - SAM: ~23ms (FastSAM-s, CUDA)
  - BLIP: ~700ms (per full image) / ~1.5s (3 crops)
  - Ollama: ~2-3s (qwen2.5:7b, 2 calls: OCR check + description)
  - TTS: ~300ms (Edge TTS EmelNeural)
- Quality: brand names identified (Kreon=sindirim ilacı), proactive OCR for medicine boxes
- Context: last 3 exchanges per WebSocket session, reset on new scene intent

## Current Latency Profile (measured 2026-03-26)
- SAM: ~60-350ms (FastSAM-s, CUDA — varies with image complexity)
- BLIP: ~920-1012ms (4 crops per image)
- Ollama: ~1129-1381ms (qwen2.5:7b, 2 calls)
- TTS: ~1087-1579ms (Edge TTS — streaming reduces perceived latency)
- **Total: ~3.2–4.6s** end-to-end

## Immediate Next Steps
1. Android frontend integration (Phase 6) — when ready
2. Phase 6 (Binary Protocol STT): implement when Android sends audio+image frames
3. EasyOCR integration (Phase 7) — after Android connected

## Open Decisions
- Android protocol extension for audio (STT via phone mic): deferred to Phase 6
- EasyOCR integration: deferred to Phase 7 (after current pipeline perfected)
- Cloud platform: RunPod or Vast.ai, planned for presentation period
