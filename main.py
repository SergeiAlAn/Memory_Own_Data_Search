"""Точка входа: Telegram-бот v2 (Haystack + Docling + Pinecone)."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import telebot

# Чтобы импорты `components.*` / `bot.*` / `pipelines.*` работали при запуске main.py
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.handlers import register_handlers  # noqa: E402
from components.assistant import AssistantService  # noqa: E402
from components.config import load_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("own_data_search")


def main() -> None:
    settings = load_settings()
    bot = telebot.TeleBot(settings.bot_token, threaded=True)
    executor = ThreadPoolExecutor(max_workers=4)

    logger.info("Инициализация AssistantService...")
    assistant = AssistantService(settings)
    logger.info(
        "Готов: index=%s namespace=%s",
        assistant.index_name,
        assistant.namespace,
    )

    register_handlers(bot, assistant, settings, executor)

    logger.info("Бот запущен (polling), timeout ask=%ss", settings.ask_timeout_sec)
    try:
        bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=50)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
