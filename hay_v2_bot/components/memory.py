"""Долговременная память: факты о пользователе в Pinecone."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from components.config import Settings
from components.store import VectorStack, user_filter

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """Извлеки из сообщения пользователя факты о НЁМ для долговременной памяти.

Что НУЖНО сохранять (это факты):
- интересы и увлечения («меня интересует Плутон», «люблю прогулки»);
- желания и цели («хочу на Марс», «хочу выучить Python»);
- имя, город, работа, предпочтения, важные события;
- формулируй от третьего лица: «Интересуется Плутоном», «Хочет на Марс».

Что НЕ сохранять:
- чистые вопросы без новых фактов;
- приветствия/команды без содержания;
- просьбы к боту (погода, факт о кошках/собаках, картинка собаки) без личных фактов;
- просьбы вспомнить уже сказанное без новой информации;
- содержимое загруженных файлов (оно уже в векторной базе отдельно).

Если в одном сообщении есть и вопрос, и факт — сохрани только факт.
Верни ТОЛЬКО JSON-массив строк. Если фактов нет — [].

Примеры:
"Меня зовут Анна, живу в Казани" → ["Пользователя зовут Анна", "Живёт в Казани"]
"Привет! Я хочу на Марс!" → ["Хочет на Марс"]
"А где я живу?" → []
"Какая погода в Москве?" → []
"Расскажи факт о кошках" → []
"Запомни: я живу на Венере" → ["Живёт на Венере"]
"""


class MemoryService:
    """Извлечение, запись и чтение фактов пользователя."""

    def __init__(self, settings: Settings, stack: VectorStack) -> None:
        self.settings = settings
        self.stack = stack
        self.fact_llm = OpenAIChatGenerator(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.chat_model,
            generation_kwargs={"temperature": 0, "max_tokens": 400},
        )

    def retrieve_context(
        self,
        query: str,
        *,
        user_id: str | int,
        top_k: int | None = None,
    ) -> list[Document]:
        embedding = self.stack.text_embedder.run(text=query)["embedding"]
        result = self.stack.retriever.run(
            query_embedding=embedding,
            filters=user_filter(user_id),
            top_k=top_k or self.settings.retrieval_top_k,
        )
        documents = result.get("documents") or []
        return [
            doc
            for doc in documents
            if (doc.score is None) or (doc.score >= self.settings.similarity_threshold)
        ]

    def load_all_facts(self, user_id: str | int) -> list[Document]:
        try:
            docs = self.stack.document_store.filter_documents(
                filters=user_filter(user_id)
            )
        except Exception:
            logger.exception("Не удалось прочитать факты user=%s", user_id)
            return []

        result: list[Document] = []
        for doc in docs:
            if not (doc.content or "").strip():
                continue
            kind = (doc.meta or {}).get("kind")
            role = (doc.meta or {}).get("role")
            if role == "assistant" or kind in {"assistant", "user_message", "document_chunk"}:
                continue
            result.append(doc)
        result.sort(key=lambda d: (d.meta or {}).get("created_at") or "", reverse=True)
        return result

    def build_context_documents(
        self,
        query: str,
        *,
        user_id: str | int,
    ) -> list[Document]:
        """Семантический поиск (факты + чанки файлов) + полный список фактов."""
        semantic = self.retrieve_context(query, user_id=user_id)
        all_facts = self.load_all_facts(user_id)

        merged: list[Document] = []
        seen: set[str] = set()

        def _add(doc: Document) -> None:
            text = (doc.content or "").strip()
            if not text or text in seen:
                return
            kind = (doc.meta or {}).get("kind")
            if kind == "user_message":
                return
            seen.add(text)
            merged.append(doc)

        for doc in semantic:
            _add(doc)
        for doc in all_facts:
            _add(doc)

        return merged[:24]

    @staticmethod
    def format_context(documents: list[Document]) -> str:
        if not documents:
            return "Пока в памяти пусто."
        lines: list[str] = []
        for i, doc in enumerate(documents, start=1):
            meta = doc.meta or {}
            kind = meta.get("kind", "user_fact")
            score = doc.score
            score_txt = f"{score:.3f}" if isinstance(score, float) else "n/a"
            header = f"[{i}] kind={kind} cosine={score_txt}"
            if kind == "document_chunk":
                file_name = meta.get("file_name", "?")
                chunk_index = meta.get("chunk_index", "?")
                page = meta.get("page_number")
                page_txt = f" page={page}" if page is not None else ""
                header += f" file={file_name} chunk={chunk_index}{page_txt}"
            lines.append(f"{header}\n{doc.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _is_savable_fact(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned or cleaned.startswith("/"):
            return False
        if "?" in cleaned or "？" in cleaned:
            return False
        return True

    def extract_facts(self, user_text: str) -> list[str]:
        cleaned = (user_text or "").strip()
        if not cleaned or cleaned.startswith("/"):
            return []

        result = self.fact_llm.run(
            messages=[
                ChatMessage.from_system(EXTRACT_PROMPT),
                ChatMessage.from_user(cleaned),
            ]
        )
        replies = result.get("replies") or []
        raw = (replies[0].text if replies else "") or ""
        raw = raw.strip()
        try:
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)
            if isinstance(data, list):
                facts = [str(item).strip() for item in data if str(item).strip()]
                return [fact for fact in facts if self._is_savable_fact(fact)]
            logger.warning("extract: unexpected JSON type=%s raw=%r", type(data).__name__, raw)
        except json.JSONDecodeError:
            logger.warning("extract: bad JSON raw=%r", raw)
        return []

    def remember_fact(self, fact: str, *, user_id: str | int) -> dict[str, Any]:
        cleaned = (fact or "").strip()
        if not cleaned:
            return {"action": "skipped", "reason": "empty"}

        nearest = self.retrieve_context(cleaned, user_id=user_id, top_k=1)
        # Дедуп только среди фактов, не среди чанков файлов
        nearest = [
            doc
            for doc in nearest
            if (doc.meta or {}).get("kind", "user_fact") == "user_fact"
        ]
        score = float(nearest[0].score) if nearest and nearest[0].score is not None else None

        if nearest and score is not None and score >= self.settings.fact_dedup_threshold:
            existing = nearest[0]
            doc_id = existing.id or f"{user_id}-user_fact-{uuid.uuid4().hex[:12]}"
            meta = dict(existing.meta or {})
            meta.update(
                {
                    "user_id": str(user_id),
                    "kind": "user_fact",
                    "role": "user",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            document = Document(id=doc_id, content=cleaned[:3500], meta=meta)
            action = "updated"
        else:
            doc_id = f"{user_id}-user_fact-{uuid.uuid4().hex[:12]}"
            meta = {
                "user_id": str(user_id),
                "kind": "user_fact",
                "role": "user",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            document = Document(id=doc_id, content=cleaned[:3500], meta=meta)
            action = "created"

        embedded = self.stack.doc_embedder.run(documents=[document])["documents"]
        self.stack.document_store.write_documents(
            embedded,
            policy=DuplicatePolicy.OVERWRITE,
        )
        logger.info("MEMORY action=%s score=%s id=%s text=%r", action, score, doc_id, cleaned)
        return {"action": action, "id": doc_id, "score": score, "text": cleaned}

    def save_user_facts(self, user_text: str, *, user_id: str | int) -> list[dict[str, Any]]:
        facts = self.extract_facts(user_text)
        if not facts:
            logger.info("MEMORY nothing to save for user=%s: %r", user_id, user_text)
            return []
        return [self.remember_fact(fact, user_id=user_id) for fact in facts]

    def forget_user(self, user_id: str | int) -> int:
        before = self.stack.document_store.count_documents()
        self.stack.document_store.delete_by_filter(filters=user_filter(user_id))
        after = self.stack.document_store.count_documents()
        return max(0, before - after)

    def list_memories(self, user_id: str | int, query: str = "обо мне") -> list[str]:
        docs = self.load_all_facts(user_id)
        if not docs:
            docs = [
                d
                for d in self.retrieve_context(query, user_id=user_id, top_k=15)
                if (d.meta or {}).get("kind") == "user_fact"
            ]

        facts: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            text = (doc.content or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            facts.append(text)
        return facts
