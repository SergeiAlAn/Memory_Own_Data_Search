"""Ingestion pipeline: Docling → enrich meta → embed → Pinecone."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from components.enricher import DocumentMetaEnricher
from components.store import VectorStack

logger = logging.getLogger(__name__)


def _load_docling_converter():
    """Импорт DoclingConverter из доступного пакета."""
    try:
        from haystack_integrations.components.converters.docling import (  # type: ignore
            DoclingConverter,
            ExportType,
        )

        return DoclingConverter, ExportType
    except ImportError:
        from docling_haystack.converter import DoclingConverter, ExportType  # type: ignore

        return DoclingConverter, ExportType


def _build_chunker():
    try:
        from docling.chunking import HybridChunker

        return HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=512,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "HybridChunker недоступен (%s), используем default Docling chunker",
            exc,
        )
        return None


@dataclass
class IngestionResult:
    documents: list[Document]
    documents_written: int
    file_name: str


class IngestionPipeline:
    """
    Пайплайн индексации файла:

    DoclingConverter (DOC_CHUNKS)
      → DocumentMetaEnricher (file_name, chunk_index, page_number, user_id)
      → OpenAIDocumentEmbedder
      → DocumentWriter → Pinecone
    """

    def __init__(self, stack: VectorStack) -> None:
        DoclingConverter, ExportType = _load_docling_converter()
        chunker = _build_chunker()

        converter_kwargs: dict = {"export_type": ExportType.DOC_CHUNKS}
        if chunker is not None:
            converter_kwargs["chunker"] = chunker

        self.converter = DoclingConverter(**converter_kwargs)
        self.enricher = DocumentMetaEnricher()
        self.embedder = stack.doc_embedder
        self.writer = DocumentWriter(
            document_store=stack.document_store,
            policy=DuplicatePolicy.OVERWRITE,
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component("enricher", self.enricher)
        self.pipeline.add_component("embedder", self.embedder)
        self.pipeline.add_component("writer", self.writer)
        self.pipeline.connect("enricher.documents", "embedder.documents")
        self.pipeline.connect("embedder.documents", "writer.documents")

        if hasattr(self.converter, "warm_up"):
            self.converter.warm_up()

    def _convert(self, path: Path) -> list[Document]:
        # Актуальный API Haystack: sources=...
        try:
            result = self.converter.run(sources=[str(path)])
            documents = result.get("documents") or []
            if documents:
                return documents
        except TypeError:
            pass

        # Совместимость со старым API docling-haystack: paths=...
        result = self.converter.run(paths=[str(path)])
        documents = result.get("documents") or []
        if not documents:
            raise RuntimeError("Docling не вернул ни одного чанка документа")
        return documents

    def run(
        self,
        file_path: str | Path,
        *,
        user_id: str | int,
        file_name: str,
    ) -> IngestionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        logger.info(
            "Ingestion start user=%s file=%s path=%s",
            user_id,
            file_name,
            path,
        )

        raw_documents = self._convert(path)
        pipe_result = self.pipeline.run(
            {
                "enricher": {
                    "documents": raw_documents,
                    "user_id": str(user_id),
                    "file_name": file_name,
                }
            },
            include_outputs_from={"enricher"},
        )

        written = int((pipe_result.get("writer") or {}).get("documents_written") or 0)
        enriched = list((pipe_result.get("enricher") or {}).get("documents") or [])
        if not enriched:
            # fallback, если include_outputs_from недоступен
            enriched = self.enricher.run(
                documents=raw_documents,
                user_id=str(user_id),
                file_name=file_name,
            )["documents"]

        logger.info(
            "Ingestion done user=%s file=%s chunks=%s written=%s",
            user_id,
            file_name,
            len(enriched),
            written,
        )
        return IngestionResult(
            documents=enriched,
            documents_written=written or len(enriched),
            file_name=file_name,
        )
