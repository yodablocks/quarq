"""RAG query, corpus management, and status endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request

from quarq.api.models import (
    RAGAddRequest,
    RAGAddResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceCitation,
)
from quarq.rag.embedder import Embedder
from quarq.rag.generator import answer
from quarq.rag.loader import load_folder, load_pdf
from quarq.rag.retriever import Retriever
from quarq.rag.store import VectorStore

router = APIRouter()

_COLLECTION_NAME = "quarq_rag_v1"


@router.post("/query", response_model=RAGQueryResponse)
def post_query(body: RAGQueryRequest, request: Request) -> RAGQueryResponse:
    """Answer a question using RAG retrieval over the indexed corpus.

    Args:
        body: RAGQueryRequest with question, optional doc_type filter, k, context.
        request: FastAPI Request (provides app.state.config).

    Returns:
        RAGQueryResponse with answer text and source citations.

    Raises:
        HTTPException 422: If the corpus is empty.
    """
    cfg = request.app.state.config
    store = VectorStore(cfg)

    if store.count() == 0:
        raise HTTPException(
            status_code=422,
            detail="RAG corpus is empty. Index documents first via POST /rag/add",
        )

    embedder = Embedder(model_name=cfg.embedder.model)
    retriever = Retriever(store=store, embedder=embedder, cfg=cfg)
    chunks = retriever.retrieve(body.question, k=body.k, doc_type=body.doc_type)

    result = answer(body.question, chunks, cfg, portfolio_context=body.portfolio_context)

    sources = [
        SourceCitation(source=c.source, page=c.page, similarity=c.similarity)
        for c in chunks
    ]

    return RAGQueryResponse(
        question=body.question,
        answer=result.answer,
        sources=sources,
        model=result.model,
        backend=result.backend,
        latency_ms=result.latency_ms,
    )


@router.post("/add", response_model=RAGAddResponse)
def post_add(body: RAGAddRequest, request: Request) -> RAGAddResponse:
    """Index a PDF file or folder of PDFs into the RAG corpus.

    Args:
        body: RAGAddRequest with path and recursive flag.
        request: FastAPI Request (provides app.state.config).

    Returns:
        RAGAddResponse with chunk counts before and after indexing.

    Raises:
        HTTPException 404: If path does not exist.
        HTTPException 422: If path contains no PDF files.
    """
    cfg = request.app.state.config
    target = Path(body.path).expanduser().resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {body.path}")

    if target.is_file():
        documents = load_pdf(
            target, chunk_size=cfg.rag.chunk_size, chunk_overlap=cfg.rag.chunk_overlap
        )
    else:
        documents = load_folder(
            target, chunk_size=cfg.rag.chunk_size, chunk_overlap=cfg.rag.chunk_overlap
        )

    if not documents:
        raise HTTPException(
            status_code=422,
            detail=f"No PDF documents found at path: {body.path}",
        )

    embedder = Embedder(model_name=cfg.embedder.model)
    store = VectorStore(cfg)

    embeddings = embedder.embed([doc.content for doc in documents])
    added = store.upsert(documents, embeddings)
    skipped = len(documents) - added

    return RAGAddResponse(
        path=str(target),
        chunks_added=added,
        chunks_skipped=skipped,
        total_chunks=store.count(),
        total_docs=store.count_sources(),
    )


@router.get("/status")
def get_rag_status(request: Request) -> dict:
    """Return corpus statistics.

    Args:
        request: FastAPI Request (provides app.state.config).

    Returns:
        Dict with chunk_count, doc_count, collection, chroma_path.
    """
    cfg = request.app.state.config
    store = VectorStore(cfg)
    return {
        "chunk_count": store.count(),
        "doc_count": store.count_sources(),
        "collection": _COLLECTION_NAME,
        "chroma_path": cfg.rag.chroma_path,
    }


@router.delete("/corpus")
def delete_corpus(
    request: Request,
    x_confirm_delete: str | None = Header(default=None, alias="X-Confirm-Delete"),
) -> dict:
    """Clear the entire RAG corpus (destructive, requires confirmation header).

    Args:
        request: FastAPI Request (provides app.state.config).
        x_confirm_delete: Must be "true" to proceed (X-Confirm-Delete header).

    Returns:
        Dict with status and collection name.

    Raises:
        HTTPException 400: If X-Confirm-Delete header is missing or not "true".
    """
    if x_confirm_delete != "true":
        raise HTTPException(
            status_code=400,
            detail="Destructive operation requires header: X-Confirm-Delete: true",
        )

    cfg = request.app.state.config
    import chromadb

    chroma_path = Path(cfg.rag.chroma_path).expanduser()
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return {"status": "cleared", "collection": _COLLECTION_NAME}
