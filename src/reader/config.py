from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str
    openai_api_key: str
    allowed_user_ids: Annotated[list[int], NoDecode] = []

    data_dir: Path = Path("data")
    ocr_model: str = "gpt-4.1-mini"
    cleanup_model: str = "gpt-4.1-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "sage"

    inline_text_max_chars: int = 1500
    tts_chunk_chars: int = 4000
    tts_part_chars: int = 25000
    tts_concurrency: int = 4
    ocr_concurrency: int = 3
    audio_bitrate: str = "48k"
    retention_days: int = 14

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def _parse_ids(cls, value: object) -> object:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            stripped = value.strip().strip("[]")
            return [int(part) for part in stripped.split(",") if part.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
