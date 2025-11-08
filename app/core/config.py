from pydantic_settings import BaseSettings
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ExtractionMode(str, Enum):
    LLM_ONLY = "llm_only"
    LLM_ENHANCED = "llm_enhanced"
    RAG_ONLY = "rag_only"
    RAG_ENHANCED = "rag_enhanced"


class Settings(BaseSettings):
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Data directories
    curricula_dir: str = "data/curricula"
    cache_dir: str = "data/cache"
    outputs_dir: str = "data/outputs"
    analysis_dir: str = "data/analysis"

    # Scraping settings
    ebooks_base_url: str = "https://ebooks.edu.gr/ebooks/v2/ps.jsp"
    max_concurrent_downloads: int = 5
    request_delay: float = 1.0

    class Config:
        env_file = ".env"


settings = Settings()
