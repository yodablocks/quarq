"""quarq RAG (retrieval-augmented generation) layer."""

from quarq.rag.embedder import Embedder
from quarq.rag.generator import answer, format_citations
from quarq.rag.loader import Document, load_folder, load_pdf
from quarq.rag.retriever import Retriever
from quarq.rag.store import RetrievedChunk, VectorStore

__all__ = [
    "Embedder",
    "answer",
    "format_citations",
    "Document",
    "load_folder",
    "load_pdf",
    "Retriever",
    "RetrievedChunk",
    "VectorStore",
]
