# RAG Pipeline

A Retrieval-Augmented Generation (RAG) backend built with **FastAPI** and the **Mistral AI API**, backed by a fully custom retrieval engine (no LangChain, no vector databases, no external RAG libraries).

---

## System Design

### Ingestion Pipeline

```
  PDF files (1–20)
        │
        ▼
  ┌─────────────────────────────────────┐
  │  PDF Parser  (pdfminer.six)         │
  │  · extract text page by page        │
  │  · sentence-aware sliding window    │
  │  · discard chunks < 50 chars        │
  └───────────────┬─────────────────────┘
                  │  chunks [ {text, file, page} ]
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
  ┌───────────────┐   ┌───────────────────┐
  │  Embedder     │   │  BM25 Index       │
  │  mistral-embed│   │  (from scratch)   │
  │  batched ×16  │   │  inverted index   │
  └───────┬───────┘   └───────────────────┘
          │  float32 vectors (L2-normalised)
          ▼
  ┌───────────────────┐
  │  Vector Store     │
  │  numpy matrix     │
  │  cosine via dot · │
  └───────────────────┘
```

### Query Flow

```
  User Query
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  Intent Detector                             │
  │  1. regex heuristics (fast, no API call)     │
  │  2. Mistral classifier (if inconclusive)     │
  └──────┬──────────────┬───────────────┬────────┘
         │              │               │
   conversational    refused      factual / list
         │              │          / comparison
         ▼              ▼               │
   direct reply    refusal msg          │
                                        ▼
                          ┌──────────────────────────┐
                          │  Query Transformer       │
                          │  1. Rewrite — explicit,  │
                          │     self-contained query │
                          │  2. HyDE — hypothetical  │
                          │     document excerpt     │
                          └────────────┬─────────────┘
                                       │  transformed query
                             ┌─────────┴──────────┐
                             │                    │
                             ▼                    ▼
                     ┌──────────────┐    ┌──────────────┐
                     │ Semantic     │    │ BM25         │
                     │ Search       │    │ Search       │
                     │ (cosine sim) │    │ (keyword)    │
                     └──────┬───────┘    └──────┬───────┘
                            │                   │
                            └─────────┬─────────┘
                                      │  Reciprocal Rank Fusion
                                      ▼
                               top-k chunks
                                      │
                                      ▼
                     ┌─────────────────────────────────┐
                     │  Generator                      │
                     │  · evidence threshold check     │
                     │  · prompt template by intent    │
                     │  · Mistral chat completion      │
                     │  · hallucination filter         │
                     │    (per-sentence evidence check)│
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                            Answer  +  Citations
```

---

## Key Design Decisions

### PDF Chunking
**pdfminer.six** is used for extraction because it handles ligatures, encoding, and multi-column PDFs more robustly than PyPDF2. Chunking is **sentence-aware with sliding window overlap**: sentences are split first via regex, then grouped greedily until ~800 chars, and the window backtracks by ~150 chars before starting the next chunk. This prevents cutting mid-thought. Very short chunks (<50 chars) are discarded — they are usually page numbers or headers that add retrieval noise.

### Embedding & Vector Store (no third-party DB)
Embeddings are obtained from Mistral's `mistral-embed` model in batches of 16. All vectors are stored in a **numpy float32 matrix**, L2-normalised at insertion so cosine similarity reduces to a fast dot product: `scores = matrix @ query_vec`. Top-k selection uses `np.argpartition` for O(N) complexity vs O(N log N) for a full sort.

### BM25 (no external library)
Classic **Okapi BM25** (k1=1.5, b=0.75) is implemented from scratch with an inverted index. Tokenisation: lowercase, strip punctuation, remove stopwords. IDF values are lazily cached and cleared on index mutation. On file deletion, the index is rebuilt in-place from remaining chunks (infrequent operation, acceptable cost).

### Hybrid Search + RRF
**Reciprocal Rank Fusion** (RRF, k=60) merges semantic and keyword ranked lists without normalising scores across two different distributions — a common problem with score-based fusion. Each retrieval set uses 2×top_k candidates before fusion, then the final top-k is returned.

### Query Transformation
Step 1 (Rewrite): Mistral rewrites the query to be explicit and self-contained (expands pronouns, acronyms, adds domain keywords). Step 2 (HyDE): Mistral generates a short hypothetical document excerpt. This is appended to the rewritten query before embedding, pulling the query vector closer to the space of actual document vectors.

### Intent Detection
Fast heuristics run first (regex patterns for greetings, PII, medical/legal phrases) to avoid LLM latency on trivially obvious cases. If heuristics are inconclusive, Mistral `mistral-small-latest` classifies into: `factual | list | comparison | conversational | refused`.

### Generation & Safety
The **evidence threshold** check ensures the system returns "insufficient evidence" rather than hallucinating when no retrieved chunk exceeds the similarity floor (default 0.35). A **hallucination filter** independently verifies each sentence in the generated answer against the context via a second Mistral call, removing sentences that can't be corroborated. **Query refusal policies** handle PII lookups, medical advice, and legal advice before retrieval is even attempted.

---

## Project Structure

```
.
├── backend/
│   ├── main.py                   # FastAPI app, CORS, static file serving
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   ├── routers/
│   │   ├── ingestion.py          # POST /api/ingest, GET /api/ingest/status
│   │   └── query.py              # POST /api/query
│   └── services/
│       ├── pdf_parser.py         # Text extraction + sentence-aware chunking
│       ├── embedder.py           # Mistral embeddings API client (batched)
│       ├── vector_store.py       # In-memory cosine search (numpy, thread-safe)
│       ├── bm25.py               # BM25 keyword search (from scratch)
│       ├── retriever.py          # Hybrid search (semantic + BM25 + RRF)
│       ├── intent_detector.py    # Intent classification + refusal policies
│       ├── query_transformer.py  # Query rewriting + HyDE
│       └── generator.py          # LLM generation + hallucination filter
└── frontend/
    └── index.html                # Single-file SPA (vanilla JS, no frameworks)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/ingest` | Upload 1–20 PDF files for ingestion |
| `GET`  | `/api/ingest/status` | Current KB state (file list, chunk count) |
| `DELETE` | `/api/ingest/{filename}` | Remove a document from the KB |
| `POST` | `/api/query` | Submit a question, get answer + citations |
| `GET`  | `/api/health` | Health check |

### POST `/api/ingest`
```
multipart/form-data  →  files: PDF files (1-20)

Response:
{
  "message": "Ingestion complete.",
  "files_processed": 2,
  "total_chunks": 143,
  "details": [{"file": "doc.pdf", "status": "ok", "chunks": 143}]
}
```

### POST `/api/query`
```json
Request:
{
  "question": "What are the key findings?",
  "top_k": 5,
  "similarity_threshold": 0.35
}

Response:
{
  "answer": "The key findings are… [Source: doc.pdf, p.3]",
  "intent": "factual",
  "citations": [
    { "source_file": "doc.pdf", "page": 3, "chunk_index": 12,
      "score": 0.82, "text_snippet": "…" }
  ],
  "query_used": "<transformed query that was actually searched>",
  "sufficient_evidence": true
}
```

---

## How to Run

### 1. Install dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure the API key

Export directly:
```bash
export MISTRAL_API_KEY=your_key_here
```

### 3. Start the server

```bash
# From the project root (one level above backend/):
uvicorn backend.main:app --reload --port 8000
```

### 4. Open the UI

Go to [http://localhost:8000](http://localhost:8000).  
FastAPI interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Libraries Used

| Library | Purpose | Link |
|---------|---------|------|
| FastAPI | Web framework | https://fastapi.tiangolo.com |
| Uvicorn | ASGI server | https://www.uvicorn.org |
| pdfminer.six | PDF text extraction | https://pdfminersix.readthedocs.io |
| NumPy | Vector math for cosine similarity | https://numpy.org |
| httpx | HTTP client for Mistral API calls | https://www.python-httpx.org |
| Mistral AI API | Embeddings + LLM generation | https://docs.mistral.ai |

> **No LangChain, no vector database, no external RAG framework.** All retrieval logic — cosine search, BM25, RRF fusion — is implemented from scratch.

---

## Bonus Features

- **Citations required**: Returns "insufficient evidence" if top-k chunk scores fall below `similarity_threshold`.
- **Answer shaping**: Separate prompt templates for `factual`, `list`, and `comparison` intents guide structured output.
- **Hallucination filter**: Post-hoc per-sentence evidence check using a second Mistral call removes unsupported claims.
- **Query refusal policies**: PII, medical advice, and legal advice requests are refused before any retrieval runs.
- **No third-party vector database**: All vectors stored in-process as a numpy float32 matrix.
