"""Generation pipeline: retrieve → Agent (+ tools) → ответ (+ сохранение фактов)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from components.config import Settings
from components.memory import MemoryService
from components.tools import build_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты умный персональный помощник в Telegram.

## Стиль
- Общайся как реальный помощник: тепло, по делу, на языке пользователя.
- Блок «Контекст из памяти» содержит:
  1) факты о пользователе из Pinecone (cosine);
  2) релевантные фрагменты загруженных файлов (kind=document_chunk).
- Считай факты о пользователе достоверными (имя, город, учёба, интересы).
- Фрагменты файлов используй для ответов по содержимому документов; при возможности
  упоминай имя файла / страницу, если они есть в контексте.
- Если спрашивают «что ты помнишь / где я живу / как меня зовут» —
  отвечай по фактам явно; не говори, что ничего не помнишь, пока факты есть.
- Не выдумывай факты о пользователе и содержимое файлов, которых нет в контексте.

## Инструменты
- `get_random_cat_fact` — случайный факт о кошках (catfact.ninja).
- `get_random_dog_fact` — случайный факт о собаках (dogapi.dog).
- `get_random_dog_image_and_describe` — картинка собаки + vision-описание породы.
- `get_current_weather` — текущая погода в городе через Open-Meteo.

Вызывай нужный tool по запросу. Для обычного разговора и вопросов по файлам tools не нужны.

Важно: картинку бот отправит в Telegram сам. В ответе не вставляй markdown-изображения
и не дублируй длинный URL — достаточно краткого описания породы и истории.
"""


@dataclass
class AssistantReply:
    text: str
    image_url: Optional[str] = None


class GenerationPipeline:
    """
    Пайплайн ответа на текстовое сообщение:

    1) Retriever ищет релевантные факты и чанки файлов пользователя;
    2) Agent (LLM + tools) формирует ответ с учётом контекста;
    3) из сообщения извлекаются новые факты о пользователе → Pinecone.
    """

    def __init__(self, settings: Settings, memory: MemoryService) -> None:
        self.settings = settings
        self.memory = memory
        self.last_image_url: Optional[str] = None

        openai_secret = Secret.from_token(settings.openai_api_key)
        self.tools = build_tools(on_image_ready=self._set_last_image_url)
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                api_key=openai_secret,
                api_base_url=settings.openai_base_url,
                model=settings.chat_model,
                generation_kwargs={"temperature": 0.5, "max_tokens": 1200},
            ),
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            max_agent_steps=8,
        )

    def _set_last_image_url(self, url: str) -> None:
        self.last_image_url = url

    def run(self, message: str, *, user_id: str | int) -> AssistantReply:
        self.last_image_url = None
        user_text = (message or "").strip()
        if not user_text:
            return AssistantReply(text="Напиши сообщение — я на связи.")

        # 1. Retriever: факты + чанки документов
        context_docs = self.memory.build_context_documents(user_text, user_id=user_id)
        context_block = self.memory.format_context(context_docs)
        logger.info(
            "Generation context user=%s docs=%s",
            user_id,
            len(context_docs),
        )

        # 2. LLM / Agent
        user_payload = (
            "Контекст из памяти (Pinecone) — факты о пользователе и фрагменты файлов:\n"
            f"{context_block}\n\n"
            f"Сообщение пользователя:\n{user_text}"
        )
        result = self.agent.run(messages=[ChatMessage.from_user(user_payload)])
        last_message = result.get("last_message")
        answer = (getattr(last_message, "text", None) or "").strip()
        if not answer:
            answer = "Готово, но текст ответа пустой. Попробуй переформулировать."

        # 3. Сохраняем только факты о пользователе
        self.memory.save_user_facts(user_text, user_id=user_id)

        return AssistantReply(text=answer, image_url=self.last_image_url)
