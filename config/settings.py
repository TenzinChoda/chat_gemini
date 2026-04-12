"""
Central settings for BT chatbot.

Production practice:
  - Tune behaviour via environment variables (prefix BTL_) or a .env file next to cwd.
  - Never commit secrets; use deployment env vars / secrets manager.
  - Defaults match local development under the bt_chatbot/ folder.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunable paths and model names."""

    model_config = SettingsConfigDict(
        env_prefix="BTL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Optional absolute overrides (set in production if data lives outside repo) ---
    project_root: Path = Field(default_factory=_default_project_root)

    # If unset, derived from project_root (see validator below)
    data_raw_dir: Path | None = None
    chroma_persist_dir: Path | None = None
    conversations_db_path: Path | None = None

    # --- Models ---
    ollama_model: str = "llama3"
    ollama_temperature: float = 0.2
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection_name: str = "bt_tickets"
    rag_retriever_k: int = 3

    # --- API / CORS (production: set CORS to your frontend origin) ---
    cors_allow_origins: str = "*"

    @model_validator(mode="after")
    def _fill_paths(self) -> Settings:
        root = self.project_root
        if self.data_raw_dir is None:
            object.__setattr__(self, "data_raw_dir", root / "data" / "raw")
        if self.chroma_persist_dir is None:
            object.__setattr__(self, "chroma_persist_dir", root / "database" / "chroma_db")
        if self.conversations_db_path is None:
            object.__setattr__(self, "conversations_db_path", root / "database" / "conversations.db")
        return self

    @computed_field
    @property
    def dir_json(self) -> Path:
        return self.data_raw_dir / "json"

    @computed_field
    @property
    def dir_excel(self) -> Path:
        return self.data_raw_dir / "excel"

    @computed_field
    @property
    def dir_csv(self) -> Path:
        return self.data_raw_dir / "csv"

    @computed_field
    @property
    def dir_pdf(self) -> Path:
        return self.data_raw_dir / "pdf"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (safe for FastAPI dependency injection)."""
    return Settings()
