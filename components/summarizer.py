"""Краткое резюме загруженного файла — ровно одно предложение."""

from __future__ import annotations

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from components.config import Settings

SUMMARY_PROMPT = """Сформулируй ровно ОДНО короткое предложение на русском языке —
краткое резюме содержимого документа для пользователя Telegram.
Без вступления, без кавычек, без списка. Только одно предложение."""


class FileSummarizer:
    def __init__(self, settings: Settings) -> None:
        self.llm = OpenAIChatGenerator(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.chat_model,
            generation_kwargs={"temperature": 0.3, "max_tokens": 120},
        )

    def summarize(self, documents: list[Document], *, file_name: str) -> str:
        snippets: list[str] = []
        total = 0
        for doc in documents:
            text = (doc.content or "").strip()
            if not text:
                continue
            piece = text[:700]
            snippets.append(piece)
            total += len(piece)
            if total >= 3500:
                break

        body = "\n\n".join(snippets) if snippets else "(пустой документ)"
        result = self.llm.run(
            messages=[
                ChatMessage.from_system(SUMMARY_PROMPT),
                ChatMessage.from_user(
                    f"Файл: {file_name}\n\nФрагменты:\n{body}"
                ),
            ]
        )
        replies = result.get("replies") or []
        text = ((replies[0].text if replies else "") or "").strip()
        if not text:
            return f"Документ «{file_name}» обработан и сохранён в базу знаний."

        # На всякий случай оставляем только первое предложение
        for sep in (". ", "! ", "? "):
            if sep in text:
                first, *_ = text.split(sep, 1)
                return (first + sep.strip()).strip()
        return text.split("\n", 1)[0].strip()
