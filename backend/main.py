from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers.ingestion import router as ingestion_router
from .routers.query import router as query_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Retrieval-Augmented Generation over PDF documents.\n\n"
        "**No external search or vector DB libraries used** — all retrieval is "
        "implemented from scratch (cosine similarity + BM25 + RRF fusion)."
    ),
    version="1.0.0",
)

# CORS — allow the frontend (served separately in dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(ingestion_router, prefix="/api")
app.include_router(query_router, prefix="/api")

# ---------------------------------------------------------------------------
# Serve static frontend (production)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["system"])
async def health():
    return {"status": "ok", "version": "1.0.0"}
