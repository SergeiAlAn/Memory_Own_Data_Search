"""Конфигурация из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str | None
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    chat_model: str
    embedding_model: str
    embedding_dimension: int
    bot_token: str
    ask_timeout_sec: int
    retrieval_top_k: int
    similarity_threshold: float
    fact_dedup_threshold: float
    tmp_dir: Path


def _require(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise ValueError(f"Не найдено ни одной из переменных: {', '.join(names)}")


def load_settings() -> Settings:
    openai_api_key = _require("OPENAI_KEY", "OPENAI_API_KEY")
    pinecone_api_key = _require("PINICONE_TOKEN", "PINECONE_API_KEY", "PINECONE_TOKEN")
    bot_token = _require("BOT_TOKEN", "TELEGRAM_BOT_TOKEN")

    # Haystack ожидает стандартные имена
    os.environ.setdefault("OPENAI_API_KEY", openai_api_key)
    os.environ.setdefault("PINECONE_API_KEY", pinecone_api_key)

    tmp_dir = ROOT_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        openai_api_key=openai_api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "own-data-search"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "own_data_search"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
        bot_token=bot_token,
        ask_timeout_sec=int(os.getenv("ASK_TIMEOUT_SEC", "150")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.12")),
        fact_dedup_threshold=float(os.getenv("FACT_DEDUP_THRESHOLD", "0.74")),
        tmp_dir=tmp_dir,
    )
