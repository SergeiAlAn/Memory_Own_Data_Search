"""Вспомогательные функции отправки сообщений в Telegram."""

from __future__ import annotations

import logging

import telebot

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Я умный персональный помощник (v2).\n\n"
    "Умею:\n"
    "• вести диалог с учётом твоего контекста (факты о тебе в Pinecone)\n"
    "• принимать файлы (PDF, DOCX и др.), разбирать через Docling и сохранять в Pinecone\n"
    "• отвечать на вопросы по загруженным документам\n"
    "• выдавать случайный факт о кошках или о собаках\n"
    "• присылать случайную картинку собаки и описывать породу через OpenAI vision\n"
    "• сообщать текущую погоду в городе (Open-Meteo)\n\n"
    "Примеры:\n"
    "• Меня зовут Алекс, я изучаю Python\n"
    "• Расскажи факт о кошках\n"
    "• Покажи случайную собаку и расскажи про породу\n"
    "• Какая погода в Москве?\n"
    "• (прикрепи PDF) — затем спроси по содержимому\n"
    "• Что ты обо мне помнишь?\n\n"
    "Команды: /start /help /memory /forget"
)

FILE_RECEIVED_TEXT = (
    "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
)
FILE_DONE_TEXT = "Готово. Я изучил этот файл, теперь можем его обсудить."


def split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def safe_send(
    bot: telebot.TeleBot,
    chat_id: int,
    text: str,
    *,
    reply_to: int | None = None,
) -> None:
    for part in split_message(text):
        try:
            bot.send_message(chat_id, part, reply_to_message_id=reply_to)
        except telebot.apihelper.ApiTelegramException:
            logger.exception("Не удалось отправить сообщение chat_id=%s", chat_id)
            try:
                bot.send_message(chat_id, part)
            except Exception:
                logger.exception("Повторная отправка тоже не удалась")


def send_photo(
    bot: telebot.TeleBot,
    chat_id: int,
    image_url: str,
    caption: str | None = None,
) -> None:
    try:
        bot.send_photo(chat_id, image_url, caption=(caption or "")[:1024] or None)
    except telebot.apihelper.ApiTelegramException:
        logger.exception("Не удалось отправить фото, шлю ссылку")
        safe_send(bot, chat_id, f"Картинка: {image_url}")
