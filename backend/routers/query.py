"""
/query endpoint — process a user question through the full RAG pipeline.

Pipeline:
    detect_intent → [refuse / converse / retrieve+generate]
                              ↓
                    transform_query
                              ↓
                    hybrid_search (semantic + BM25 + RRF)
                              ↓
                    generate_answer (evidence check + hallucination filter)
                              ↓
                    return QueryResponse with citations
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..models.schemas import QueryRequest, QueryResponse, ChunkCitation
from ..services.intent_detector import detect_intent, needs_retrieval
from ..services.query_transformer import transform_query
from ..services.retriever import hybrid_search
from ..services.generator import generate_answer, conversational_reply, refused_reply
from ..services.vector_store import store

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(req: QueryRequest) -> QueryResponse:
    """
    Process a user question against the knowledge base.

    - Intent is detected first; greetings/small-talk bypass retrieval.
    - Refused intents (PII, legal, medical) return a polite refusal.
    - For all other intents, the query is transformed, then hybrid search runs.
    - If retrieved evidence is below the similarity threshold, returns "insufficient evidence".
    """
    question = req.question.strip()

    # --- Guard: empty KB ---
    if store.total_chunks == 0 and not question:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Please ingest documents first.",
        )

    # --- Step 1: Intent detection ---
    intent = detect_intent(question)

    # --- Step 2a: Handle conversational ---
    if intent == "conversational":
        return QueryResponse(
            answer=conversational_reply(question),
            intent="conversational",
            citations=[],
            query_used=question,
            sufficient_evidence=True,
        )

    # --- Step 2b: Handle refused ---
    if intent == "refused":
        return QueryResponse(
            answer=refused_reply(question),
            intent="refused",
            citations=[],
            query_used=question,
            sufficient_evidence=False,
        )

    # --- Step 3: Empty KB check for retrieval intents ---
    if store.total_chunks == 0:
        return QueryResponse(
            answer="The knowledge base is currently empty. Please upload PDF documents first.",
            intent=intent,
            citations=[],
            query_used=question,
            sufficient_evidence=False,
        )

    # --- Step 4: Query transformation ---
    transformed_query = transform_query(question)

    # --- Step 5: Hybrid retrieval ---
    chunks = hybrid_search(
        query=transformed_query,
        top_k=req.top_k,
        semantic_k=req.top_k * 2,
        keyword_k=req.top_k * 2,
    )

    # --- Step 6: Generation with evidence check + hallucination filter ---
    answer, sufficient_evidence = generate_answer(
        question=question,
        chunks=chunks,
        intent=intent,
        similarity_threshold=req.similarity_threshold,
    )

    # --- Step 7: Build citations ---
    citations = []
    if sufficient_evidence:
        for chunk in chunks:
            citations.append(
                ChunkCitation(
                    source_file=chunk["source_file"],
                    page=chunk["page"],
                    chunk_index=chunk["chunk_index"],
                    score=round(chunk.get("score", 0.0), 4),
                    text_snippet=chunk["text"][:200] + ("…" if len(chunk["text"]) > 200 else ""),
                )
            )

    return QueryResponse(
        answer=answer,
        intent=intent,
        citations=citations,
        query_used=transformed_query,
        sufficient_evidence=sufficient_evidence,
    )
