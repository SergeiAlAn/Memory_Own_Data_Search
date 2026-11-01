"""Pinecone document store и OpenAI embedders."""

from __future__ import annotations

from dataclasses import dataclass

from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from components.config import Settings


@dataclass
class VectorStack:
    document_store: PineconeDocumentStore
    doc_embedder: OpenAIDocumentEmbedder
    text_embedder: OpenAITextEmbedder
    retriever: PineconeEmbeddingRetriever


def build_vector_stack(settings: Settings) -> VectorStack:
    openai_secret = Secret.from_token(settings.openai_api_key)
    pinecone_secret = Secret.from_token(settings.pinecone_api_key)

    document_store = PineconeDocumentStore(
        api_key=pinecone_secret,
        index=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
        show_progress=False,
    )

    doc_embedder = OpenAIDocumentEmbedder(
        api_key=openai_secret,
        api_base_url=settings.openai_base_url,
        model=settings.embedding_model,
        progress_bar=False,
    )
    text_embedder = OpenAITextEmbedder(
        api_key=openai_secret,
        api_base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )
    retriever = PineconeEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.retrieval_top_k,
    )
    return VectorStack(
        document_store=document_store,
        doc_embedder=doc_embedder,
        text_embedder=text_embedder,
        retriever=retriever,
    )


def user_filter(user_id: str | int) -> dict:
    return {
        "field": "meta.user_id",
        "operator": "==",
        "value": str(user_id),
    }
