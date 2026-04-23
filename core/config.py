"""
Application configuration loaded from environment variables or a .env file.
All configurable values should live here - never hard-code them in other modules.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    app_name: str = "SAM Assistive Vision API"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool | str = True
    model_device: str = "auto"  # "auto" | "cuda" | "cpu"
    hf_local_files_only: bool = True

    # --- Per-service mock flags (control each pipeline stage independently) ---
    # True  = use mock (fast, no model needed)
    # False = use real implementation
    mock_stt: bool = True         # Stage 1: set False when faster-whisper ready
    mock_sam: bool = True         # Stage 3: set False when FastSAM ready
    mock_blip: bool = True        # Stage 4: set False when BLIP ready
    mock_ollama: bool = True      # Stage 5: set False when Ollama running
    mock_tts: bool = True         # Stage 6: set False to use Edge TTS now

    # Mock timing (ms) — only used when the corresponding mock flag is True
    mock_stt_delay_ms: int = 150
    mock_sam_delay_ms: int = 120
    mock_blip_delay_ms: int = 600
    mock_ollama_delay_ms: int = 800
    mock_tts_delay_ms: int = 300

    # STT - faster-whisper (used when mock_stt=False)
    stt_model: str = "tiny"   # "tiny" | "base" | "small" | "medium"
    stt_min_audio_bytes: int = 8000  # Ignore very short/noisy audio envelopes
    stt_max_audio_seconds: float = 8.0
    stt_compute_type: str = "auto"  # auto -> float16 on CUDA, int8 on CPU

    # SAM (used when mock_sam=False)
    sam_model: str = "FastSAM-s.pt"   # FastSAM-s (fast GPU) or FastSAM-x (accurate)
    sam_max_crops: int = 2            # More crops improve detail but increase BLIP latency
    sam_min_area_fraction: float = 0.02

    # Caption model - HuggingFace Transformers (used when mock_blip=False)
    blip_model: str = "Salesforce/blip-image-captioning-large"
    blip_model_revision: str = ""  # unused for BLIP; kept for future model swaps
    blip_max_captions: int = 2
    blip_max_new_tokens: int = 24
    # Alternatives tried:
    #   "vikhyatk/moondream2" rev 2025-01-09 -- pyvips required (Windows: needs libvips DLL)
    #   "vikhyatk/moondream2" rev 2024-08-26 -- PhiConfig incompatible with transformers 5.x
    #   "Salesforce/blip-image-captioning-base" -- ~2GB VRAM, ~400ms

    # Ollama - LLM (used when mock_ollama=False)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Alternatives: "phi3:mini" (recommended), "mistral:7b" (best quality, slower)
    ollama_timeout_seconds: int = 30
    ollama_num_ctx: int = 512   # KV cache size; prompts use ~300 tokens so 512 is safe
    ollama_num_predict: int = 64
    ollama_temperature: float = 0.1
    ollama_keep_alive: str = "30m"
    scene_ollama_model: str = "qwen2.5:7b"
    scene_ollama_num_predict: int = 64
    scene_ollama_num_ctx: int = 512
    ocr_summary_model: str = "qwen2.5:7b"
    ocr_summary_num_predict: int = 72
    ocr_summary_num_ctx: int = 512
    ocr_backend: str = "ollama_vision"  # "paddleocr" | "ollama_vision"
    ocr_model: str = "glm-ocr:latest"
    ocr_timeout_seconds: int = 45
    paddleocr_lang: str = "tr"
    paddleocr_version: str = "PP-OCRv5"

    # Image preprocessing - resize before SAM/caption inference
    max_frame_size: int = 640  # Max dimension in px; larger frames are thumbnailed
    max_ws_frame_bytes: int = 10 * 1024 * 1024  # Protect WebSocket from oversized payloads

    # TTS — Edge TTS (used when mock_tts=False)
    tts_voice: str = "tr-TR-AhmetNeural"  # Turkish male neural voice
    tts_rate: str = "+10%"                # Slightly faster than default, more natural than +30%
    tts_streaming: bool = True            # Stream chunks over WebSocket for faster first audio

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )


# Single global instance — import this everywhere
settings = Settings()
