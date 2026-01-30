"""
RAG module for Julie - Knowledge retrieval from documents.
"""

from julie.rag.embeddings import get_embeddings
from julie.rag.store import VectorStore
from julie.rag.loader import DocumentLoader
from julie.rag.retriever import RAGRetriever

__all__ = ["get_embeddings", "VectorStore", "DocumentLoader", "RAGRetriever"]
