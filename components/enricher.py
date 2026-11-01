"""Обогащение метаданных чанков после Docling."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from haystack import Document, component


def _safe_slug(value: str, *, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return (slug or "file")[:max_len]


def extract_page_number(meta: dict[str, Any]) -> int | None:
    """Достаёт номер страницы из Docling-метаданных, если есть."""
    dl_meta = meta.get("dl_meta")
    if not isinstance(dl_meta, dict):
        return None

    try:
        doc_items = ((dl_meta.get("meta") or {}).get("doc_items")) or []
        if not doc_items:
            return None
        prov = (doc_items[0] or {}).get("prov") or []
        if not prov:
            return None
        page = (prov[0] or {}).get("page_no")
        return int(page) if page is not None else None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


@component
class DocumentMetaEnricher:
    """Добавляет user_id, file_name, chunk_index, page_number к чанкам."""

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        user_id: str,
        file_name: str,
    ) -> dict[str, list[Document]]:
        enriched: list[Document] = []
        slug = _safe_slug(file_name)
        created_at = datetime.now(timezone.utc).isoformat()

        for index, doc in enumerate(documents):
            content = (doc.content or "").strip()
            if not content:
                continue

            raw_meta = dict(doc.meta or {})
            page_number = extract_page_number(raw_meta)

            # Pinecone принимает только str/int/bool/List[str] — dl_meta и прочие
            # вложенные объекты отбрасываем после извлечения page_number.
            meta = {
                "user_id": str(user_id),
                "kind": "document_chunk",
                "role": "document",
                "file_name": file_name,
                "chunk_index": index,
                "created_at": created_at,
            }
            if page_number is not None:
                meta["page_number"] = page_number

            doc_id = (
                f"{user_id}-doc-{slug}-{index}-{uuid.uuid4().hex[:10]}"
            )
            enriched.append(
                Document(
                    id=doc_id,
                    content=content[:8000],
                    meta=meta,
                )
            )

        return {"documents": enriched}
