from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Values can be loaded from .env and environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================
    # API KEYS
    # ==========================================================

    groq_api_key: str = Field(
        ...,
        alias="GROQ_API_KEY",
    )

    # ==========================================================
    # EMBEDDINGS
    # ==========================================================

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ==========================================================
    # CHUNKING
    # ==========================================================

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ==========================================================
    # RETRIEVAL
    # ==========================================================

    top_k: int = 5

    # ==========================================================
    # VECTOR DATABASE
    # ==========================================================

    collection_name: str = Field(
        default="research_paper",
        alias="COLLECTION_NAME",
    )

    # ==========================================================
    # LLM
    # ==========================================================

    llm_model: str = "llama-3.3-70b-versatile"

    temperature: float = 0.0

    max_tokens: int = 2048


settings = Settings()


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdfs"

CHROMA_DB_DIR = DATA_DIR / "chroma_db"

LOG_DIR = BASE_DIR / "logs"