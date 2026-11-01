"""Фасад: собирает store, memory, ingestion и generation в одном сервисе."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from components.config import Settings, load_settings
from components.memory import MemoryService
from components.store import build_vector_stack
from components.summarizer import FileSummarizer
from pipelines.generation import AssistantReply, GenerationPipeline
from pipelines.ingestion import IngestionPipeline, IngestionResult

logger = logging.getLogger(__name__)


class AssistantService:
    """Единая точка входа для Telegram-хендлеров."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.stack = build_vector_stack(self.settings)
        self.memory = MemoryService(self.settings, self.stack)
        self.generation = GenerationPipeline(self.settings, self.memory)
        self.ingestion = IngestionPipeline(self.stack)
        self.summarizer = FileSummarizer(self.settings)

        logger.info(
            "AssistantService ready index=%s namespace=%s metric=cosine",
            self.settings.pinecone_index_name,
            self.settings.pinecone_namespace,
        )

    @property
    def index_name(self) -> str:
        return self.settings.pinecone_index_name

    @property
    def namespace(self) -> str:
        return self.settings.pinecone_namespace

    def ask(self, message: str, *, user_id: str | int) -> AssistantReply:
        return self.generation.run(message, user_id=user_id)

    def list_memories(self, user_id: str | int) -> list[str]:
        return self.memory.list_memories(user_id)

    def forget_user(self, user_id: str | int) -> int:
        return self.memory.forget_user(user_id)

    def ingest_file(
        self,
        file_path: str | Path,
        *,
        user_id: str | int,
        file_name: str,
    ) -> IngestionResult:
        return self.ingestion.run(file_path, user_id=user_id, file_name=file_name)

    def summarize_file(self, result: IngestionResult) -> str:
        return self.summarizer.summarize(result.documents, file_name=result.file_name)

    def test_connection(self, query: str = "привет") -> dict[str, Any]:
        docs = self.memory.retrieve_context(query, user_id="connection-test", top_k=1)
        return {
            "ok": True,
            "index": self.index_name,
            "namespace": self.namespace,
            "metric": "cosine",
            "documents_in_store": self.stack.document_store.count_documents(),
            "hits": len(docs),
        }
