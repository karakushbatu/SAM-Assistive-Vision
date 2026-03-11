"""
Application configuration loaded from environment variables or a .env file.
All configurable values should live here — never hard-code them in other modules.

Per-service mock flags allow gradual rollout of real implementations:
    MOCK_SAM=true        Phase 2 not done yet  -> keep mock
    MOCK_CLASSIFIER=true Phase 3 not done yet  -> keep mock
    MOCK_OLLAMA=false    Phase 4 done          -> use real Ollama
    MOCK_TTS=false       Phase 4A done         -> use real Edge TTS
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    app_name: str = "SAM Assistive Vision API"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # --- Per-service mock flags (control each stage independently) ---
    # True  = use mock (fast, no model needed)
    # False = use real implementation
    mock_sam: bool = True         # Phase 2: set False when FastSAM ready
    mock_classifier: bool = True  # Phase 3: set False when CLIP ready
    mock_ollama: bool = True      # Phase 4B: set False when Ollama running
    mock_tts: bool = True         # Phase 4A: set False to use Edge TTS now

    # Mock timing (ms) — only used when the corresponding mock flag is True
    mock_sam_delay_ms: int = 120
    mock_classifier_delay_ms: int = 45
    mock_ollama_delay_ms: int = 800
    mock_tts_delay_ms: int = 300

    # Ollama (used when mock_ollama=False)
    ollama_base_url: str = "http://localhost:11434"  # localhost for local dev
    ollama_model: str = "llama3:8b"
    ollama_timeout_seconds: int = 30

    # TTS — Edge TTS (used when mock_tts=False)
    tts_voice: str = "tr-TR-EmelNeural"  # Turkish female voice
    tts_rate: str = "+0%"                # Speech speed: +20% for faster

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Single global instance — import this everywhere
settings = Settings()
