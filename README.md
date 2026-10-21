# ContextIQ

ContextIQ is a document question-answering app that lets you upload a PDF and ask questions about its contents. It retrieves relevant passages, generates an answer grounded in those passages, and shows the source pages so you can verify the response.

## Features

- Upload and index text-based PDF files up to 15 MB.
- Extract text page by page and split it into overlapping chunks while preserving page and chunk references.
- Create local embeddings with `sentence-transformers/all-MiniLM-L6-v2` and store them in persistent ChromaDB collections.
- Generate short descriptions of each chunk's document context through OpenRouter.
- Choose among three retrieval modes:
  - **Baseline RAG** — semantic search over the original passages.
  - **Contextual RAG** — semantic search over passages enriched with generated context.
  - **Hybrid RAG** — combines semantic and BM25 keyword rankings using Reciprocal Rank Fusion.
- Generate answers from retrieved evidence and display expandable source passages with filenames and page numbers.

## How it works

1. The backend extracts text from each PDF page and creates word-based chunks (450 words with an 80-word overlap).
2. For each chunk, it asks an OpenRouter model to describe the chunk's context, then creates embeddings for both the original and context-enriched text.
3. ChromaDB stores the baseline and contextual indexes locally in `backend/chroma_db`.
4. When you ask a question, the selected retrieval mode finds relevant passages. The backend sends those passages to OpenRouter to produce a document-grounded answer.
5. The UI displays the answer alongside the original source passages and their page numbers.

Scanned PDFs without an embedded text layer are not OCR-processed and cannot be indexed.

## Tech stack

- **Frontend:** React, Vite
- **Backend:** Python, FastAPI
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector storage:** ChromaDB
- **Keyword retrieval:** BM25 (`rank-bm25`)
- **Answer and context generation:** OpenRouter API

## Requirements

- Python 3.10 or later
- Node.js 20.19+ or 22.12+
- An OpenRouter API key

The first backend startup downloads the Sentence Transformers model if it is not already available locally. PDF context generation and question answering both require access to OpenRouter.

## Run locally

### 1. Configure the backend

From the repository root, create and activate a virtual environment:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" python-multipart chromadb pymupdf requests python-dotenv sentence-transformers rank-bm25
```

Create `backend/.env` and add your OpenRouter key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional: use the same OpenRouter model for chunk context and answers.
OPENROUTER_MODEL=your_openrouter_model_id

# Optional; defaults to the Vite development origin below.
FRONTEND_ORIGIN=http://localhost:5173
```

Start the API from the `backend` directory:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. FastAPI's interactive API documentation is at `http://localhost:8000/docs`.

### 2. Start the frontend

In a second terminal, from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend currently connects to the backend at `http://localhost:8000`.

## API overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check API status and the number of indexed chunks. |
| `POST` | `/documents/upload` | Upload and index one PDF as multipart form data under the `file` field. |
| `POST` | `/chat/ask` | Ask a question against the indexed documents. |

Example question request:

```json
{
  "question": "What are the main points?",
  "top_k": 5,
  "retrieval_mode": "hybrid"
}
```

`retrieval_mode` accepts `baseline`, `contextual`, or `hybrid`. `top_k` is constrained by the backend to a maximum of 10.

## Project structure

```text
backend/
  app/
    contextualizer.py   # Generates chunk context through OpenRouter
    main.py             # FastAPI app, PDF indexing, retrieval, and chat
  chroma_db/            # Persistent local vector database (created/updated at runtime)
  uploads/              # Temporary PDF upload storage
frontend/
  src/
    App.jsx             # Upload and document chat interface
    App.css
    index.css
    main.jsx
```

## Data and configuration notes

- Keep `backend/.env` private; it contains your OpenRouter API key.
- Indexed chunks and embeddings persist in `backend/chroma_db`. They are local application data and may need to be recreated when setting up a fresh clone.
- Uploaded PDF files are removed from `backend/uploads` after indexing; extracted chunks and metadata remain in ChromaDB.
- `OPENROUTER_MODEL` is optional. If omitted, the backend uses its built-in defaults for context generation and answer generation.
