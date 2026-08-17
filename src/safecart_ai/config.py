from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAFECART_AI_",
        extra="ignore",
    )

    env: str = "development"
    data_path: Path = Path("data/sample")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
