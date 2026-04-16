"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    total_chunks: int
    details: list[dict]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="User question or message")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of chunks to retrieve")
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0,
                                        description="Minimum similarity score for a chunk to be used")


class ChunkCitation(BaseModel):
    source_file: str
    page: int
    chunk_index: int
    score: float
    text_snippet: str


class QueryResponse(BaseModel):
    answer: str
    intent: str                         # "factual" | "conversational" | "list" | "comparison" | "refused"
    citations: list[ChunkCitation]
    query_used: str                     # the transformed query that was actually searched
    sufficient_evidence: bool


# ---------------------------------------------------------------------------
# Store info
# ---------------------------------------------------------------------------

class StoreStatus(BaseModel):
    total_documents: int
    total_chunks: int
    indexed_files: list[str]
