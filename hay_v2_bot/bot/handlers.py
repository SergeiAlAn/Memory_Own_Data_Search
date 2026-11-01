"""Обработчики Telegram: текст, команды, файлы."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import telebot
from telebot import types

from bot.messaging import (
    FILE_DONE_TEXT,
    FILE_RECEIVED_TEXT,
    HELP_TEXT,
    safe_send,
    send_photo,
    split_message,
)
from components.assistant import AssistantService
from components.config import Settings

logger = logging.getLogger(__name__)

# Расширения, которые Docling обычно умеет разбирать
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".wav",
    ".mp3",
    ".asciidoc",
    ".adoc",
}


def register_handlers(
    bot: telebot.TeleBot,
    assistant: AssistantService,
    settings: Settings,
    executor: ThreadPoolExecutor,
) -> None:
    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: types.Message) -> None:
        bot.reply_to(message, f"Привет! {HELP_TEXT}")

    @bot.message_handler(commands=["memory"])
    def handle_memory(message: types.Message) -> None:
        user_id = message.from_user.id
        bot.send_chat_action(message.chat.id, "typing")
        try:
            facts = assistant.list_memories(user_id)
        except Exception:
            logger.exception("Ошибка /memory user=%s", user_id)
            bot.reply_to(message, "Не удалось прочитать память. Попробуй позже.")
            return

        if not facts:
            bot.reply_to(
                message,
                "Пока ничего не помню. Расскажи о себе или напиши «запомни …».",
            )
            return

        lines = ["Вот что нашлось в памяти (по cosine similarity):"]
        for i, fact in enumerate(facts[:10], start=1):
            lines.append(f"{i}. {fact}")
        bot.reply_to(message, "\n".join(lines))

    @bot.message_handler(commands=["forget"])
    def handle_forget(message: types.Message) -> None:
        user_id = message.from_user.id
        try:
            removed = assistant.forget_user(user_id)
            bot.reply_to(message, f"Память очищена (удалено записей ≈ {removed}).")
        except Exception:
            logger.exception("Ошибка /forget user=%s", user_id)
            bot.reply_to(message, "Не удалось очистить память. Попробуй позже.")

    @bot.message_handler(content_types=["document"])
    def handle_document(message: types.Message) -> None:
        document = message.document
        if document is None:
            return

        user_id = message.from_user.id
        chat_id = message.chat.id
        file_name = document.file_name or f"document_{document.file_id}"
        suffix = Path(file_name).suffix.lower()

        if suffix and suffix not in SUPPORTED_SUFFIXES:
            bot.reply_to(
                message,
                f"Формат «{suffix}» может не поддерживаться Docling. "
                "Попробуй PDF или DOCX.",
            )
            # всё равно попробуем обработать ниже — Docling сам решит

        bot.reply_to(message, FILE_RECEIVED_TEXT)
        bot.send_chat_action(chat_id, "typing")

        tmp_path = settings.tmp_dir / f"{user_id}_{uuid.uuid4().hex}{suffix or '.bin'}"

        def _process() -> tuple[str, str]:
            file_info = bot.get_file(document.file_id)
            payload = bot.download_file(file_info.file_path)
            tmp_path.write_bytes(payload)

            result = assistant.ingest_file(
                tmp_path,
                user_id=user_id,
                file_name=file_name,
            )
            summary = assistant.summarize_file(result)
            return FILE_DONE_TEXT, summary

        future = executor.submit(_process)
        try:
            done_text, summary = future.result(timeout=settings.ask_timeout_sec)
            safe_send(bot, chat_id, done_text, reply_to=message.message_id)
            safe_send(bot, chat_id, summary)
        except FuturesTimeout:
            logger.error("Таймаут ingestion user=%s file=%s", user_id, file_name)
            safe_send(
                bot,
                chat_id,
                "Слишком долго обрабатываю файл. Попробуй файл поменьше или позже.",
                reply_to=message.message_id,
            )
        except Exception:
            logger.exception("Ошибка обработки файла user=%s file=%s", user_id, file_name)
            safe_send(
                bot,
                chat_id,
                "Не удалось разобрать файл. Проверь формат и попробуй ещё раз.",
                reply_to=message.message_id,
            )
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                logger.warning("Не удалось удалить tmp %s", tmp_path)

    @bot.message_handler(content_types=["text"])
    def handle_text(message: types.Message) -> None:
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return

        user_id = message.from_user.id
        chat_id = message.chat.id
        logger.info("user=%s text=%r", user_id, text[:200])

        status = None
        try:
            status = bot.reply_to(message, "Думаю: смотрю контекст и готовлю ответ…")
        except telebot.apihelper.ApiTelegramException:
            logger.exception("Не удалось отправить статус")

        bot.send_chat_action(chat_id, "typing")

        future = executor.submit(assistant.ask, text, user_id=user_id)
        image_url = None
        try:
            reply = future.result(timeout=settings.ask_timeout_sec)
            answer = reply.text
            image_url = reply.image_url
            logger.info(
                "user=%s answer_len=%s image=%s",
                user_id,
                len(answer or ""),
                bool(image_url),
            )
        except FuturesTimeout:
            logger.error("Таймаут ask() user=%s after %ss", user_id, settings.ask_timeout_sec)
            answer = (
                "Слишком долго жду ответ от модели/памяти. "
                "Попробуй ещё раз чуть позже."
            )
        except Exception:
            logger.exception("Ошибка обработки сообщения user=%s", user_id)
            answer = "Не удалось обработать сообщение. Попробуй ещё раз."

        if not answer or not str(answer).strip():
            answer = "Получил пустой ответ. Попробуй переформулировать."

        if image_url:
            send_photo(bot, chat_id, image_url, caption="Случайная собака 🐶")

        parts = split_message(str(answer))
        if status is not None:
            try:
                bot.edit_message_text(parts[0], chat_id=chat_id, message_id=status.message_id)
            except telebot.apihelper.ApiTelegramException:
                safe_send(bot, chat_id, parts[0], reply_to=message.message_id)
            for part in parts[1:]:
                safe_send(bot, chat_id, part)
        else:
            for part in parts:
                safe_send(bot, chat_id, part, reply_to=message.message_id)
