"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Text Classification Studio API"
    app_version: str = "1.0.0"

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Paths (resolved relative to backend/ unless absolute)
    artifacts_dir: str = "artifacts"
    data_dir: str = "data"

    # Prediction limits
    max_text_length: int = 5000
    max_batch_size: int = 200

    # Optional history
    enable_history: bool = True
    history_db_path: str = "artifacts/history.db"
    history_store_text: bool = False

    # Reproducibility
    random_seed: int = 42

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else BACKEND_ROOT / path

    @property
    def artifacts_path(self) -> Path:
        return self._resolve(self.artifacts_dir)

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def history_db(self) -> Path:
        return self._resolve(self.history_db_path)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
