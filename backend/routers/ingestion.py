"""
/ingest endpoint — upload one or more PDF files for processing.

Flow:
    1. Validate files are PDFs (by content-type and magic bytes).
    2. Extract text and chunk each PDF.
    3. Embed all chunks (batched Mistral API call).
    4. Add embeddings to in-memory VectorStore.
    5. Add chunk texts to BM25 index.
    6. Return ingestion summary.
"""

from __future__ import annotations

import io
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from ..models.schemas import IngestResponse
from ..services.pdf_parser import extract_chunks
from ..services.embedder import embed_texts
from ..services.vector_store import store
from ..services.bm25 import bm25_index

router = APIRouter(prefix="/ingest", tags=["ingestion"])

PDF_MAGIC = b"%PDF"
MAX_FILE_SIZE_MB = 50


@router.post("", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_documents(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
) -> IngestResponse:
    """
    Upload and ingest PDF files into the knowledge base.

    - Accepts 1–20 files per request.
    - Files are validated for PDF format.
    - Previously ingested versions of the same filename are replaced.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per request.")

    details = []
    total_chunks = 0

    for upload in files:
        filename = upload.filename or "unknown.pdf"

        # --- Read bytes ---
        raw_bytes = await upload.read()

        # --- Size guard ---
        size_mb = len(raw_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the {MAX_FILE_SIZE_MB} MB limit ({size_mb:.1f} MB).",
            )

        # --- Validate PDF magic bytes ---
        if not raw_bytes.startswith(PDF_MAGIC):
            raise HTTPException(
                status_code=415,
                detail=f"{filename} does not appear to be a valid PDF file.",
            )

        # --- Remove stale version if re-ingesting the same file ---
        store.remove_file(filename)
        bm25_index.remove_file(filename)

        # --- Extract and chunk ---
        try:
            chunks = extract_chunks(raw_bytes, filename)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse {filename}: {exc}",
            )

        if not chunks:
            details.append({"file": filename, "status": "skipped", "chunks": 0,
                             "reason": "No extractable text found."})
            continue

        # --- Embed ---
        texts = [c["text"] for c in chunks]
        try:
            embeddings = embed_texts(texts)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding API error for {filename}: {exc}",
            )

        # --- Store ---
        store.add_chunks(chunks, embeddings)
        bm25_index.add_chunks(chunks)

        details.append({"file": filename, "status": "ok", "chunks": len(chunks)})
        total_chunks += len(chunks)

    return IngestResponse(
        message="Ingestion complete.",
        files_processed=sum(1 for d in details if d["status"] == "ok"),
        total_chunks=total_chunks,
        details=details,
    )


@router.get("/status")
async def get_status():
    """Return the current state of the knowledge base."""
    return {
        "total_chunks": store.total_chunks,
        "indexed_files": store.indexed_files,
    }


@router.delete("/{filename}")
async def delete_document(filename: str):
    """Remove a document from the knowledge base by filename."""
    removed_vec = store.remove_file(filename)
    bm25_index.remove_file(filename)
    if removed_vec == 0:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in the knowledge base.")
    return {"message": f"'{filename}' removed.", "chunks_removed": removed_vec}
