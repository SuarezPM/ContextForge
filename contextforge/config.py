"""Configuration management via environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """All configuration via environment variables - no hardcoded values."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # vLLM Server
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "Qwen/Qwen3.6-35B-A3B"
    vllm_api_key: str = "contextforge-local"

    # ContextForge
    contextforge_host: str = "0.0.0.0"
    contextforge_port: int = 8001
    contextforge_ttl_seconds: int = 300
    contextforge_dedup_threshold: float = 0.85
    contextforge_compression_rate: float = 0.5
    contextforge_min_tokens_to_compress: int = 100

    # Models
    embedder_model: str = "all-MiniLM-L6-v2"
    compressor_model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

    # AMD ROCm
    rocmsmi_path: str = "/opt/rocm/bin/rocm-smi"


settings = Settings()
