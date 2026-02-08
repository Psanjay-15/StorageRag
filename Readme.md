# StorageRag

A **RAG (Retrieval-Augmented Generation)** API that indexes PDFs into [Qdrant](https://qdrant.tech/) and answers questions using [Ollama](https://ollama.ai/). PDFs are processed in memory only—nothing is written to disk.

---

## Architecture

```
┌─────────────┐     POST /upload-pdf      ┌──────────────┐
│   Client    │ ─────────────────────────►│   FastAPI    │
└─────────────┘                           └──────┬───────┘
       │                                         │
       │  POST /query (question)                 │  pdf_bytes
       │ ◄───────────────────────────────────────┤
       │         { "answer": "..." }             ▼
       │                                 ┌───────────────┐
       │                                 │  PDF Processor │  PyMuPDF
       │                                 │  (in-memory)   │
       │                                 └───────┬───────┘
       │                                         │ text
       │                                         ▼
       │                                 ┌───────────────┐
       │                                 │ Text Splitter │  LangChain
       │                                 │ (chunk 450)   │
       │                                 └───────┬───────┘
       │                                         │ chunks
       │                                         ▼
       │                                 ┌───────────────┐
       │                                 │  Embeddings   │  HuggingFace
       │                                 │ (MiniLM-L6)   │  (384-dim)
       │                                 └───────┬───────┘
       │                                         │ vectors
       │                                         ▼
       │                                 ┌───────────────┐
       │                                 │    Qdrant     │  localhost:6333
       │                                 │  (vector DB)  │
       │                                 └───────┬───────┘
       │                                         │
       │  POST /query                            │ retrieve top-k
       │ ────────────────────────────────────────┤
       │                                         ▼
       │                                 ┌───────────────┐
       │                                 │  ChatOllama   │  llama3.2:1b
       │                                 │  (context+Q)  │
       │                                 └───────┬───────┘
       │                                         │
       │         { "answer": "..." } ◄────────────┘
```

**Flow summary**

- **Upload:** PDF bytes → PyMuPDF (text) → RecursiveCharacterTextSplitter (chunks) → HuggingFace embeddings → Qdrant. One active collection per session (latest upload).
- **Query:** Question → same embeddings → Qdrant similarity search (top 5) → context + question → Ollama → answer.

---

## Prerequisites

- **Python** 3.10+
- **Qdrant** running locally (default port `6333`)
- **Ollama** running locally with model `llama3.2:1b`

---

## Setup & Run

### 1. Clone and enter the project

```bash
cd /path/to/StorageRag
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant (if not already running)

Using Docker:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Or use a [local binary](https://qdrant.tech/documentation/guides/installation/) and ensure it listens on port `6333`.

### 5. Start Ollama and pull the model

```bash
ollama serve          # if not running as a service
ollama pull llama3.2:1b
```

### 6. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: **http://localhost:8000**
- Docs: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint        | Description                    |
|--------|-----------------|--------------------------------|
| `GET`  | `/health`       | Health check and index status |
| `POST` | `/upload-pdf`   | Upload and index a PDF         |
| `POST` | `/query`        | Ask a question over the index  |

### Health

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "active_collection": "pdf_my_doc_abc12345",
  "has_active_index": true
}
```

### Upload PDF

```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@/path/to/document.pdf"
```

Example response:

```json
{
  "message": "PDF processed and indexed in Qdrant (no file saved to disk)",
  "filename": "document.pdf",
  "collection_name": "pdf_document_abc12345",
  "chunk_count": 42
}
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Example response:

```json
{
  "answer": "The document describes..."
}
```

If no PDF has been uploaded yet, `/query` returns `503` with a message asking you to upload a PDF first.

---

## Project Structure

```
StorageRag/
├── app/
│   ├── main.py           # FastAPI app: /health, /upload-pdf, /query
│   ├── models.py         # Pydantic: Query, UploadResponse
│   ├── rag_system.py     # Indexing, retriever, LLM, Qdrant + Ollama
│   └── utils/
│       └── pdf_processor.py   # PyMuPDF text extraction from bytes
├── requirements.txt
└── Readme.md
```

---

## Configuration (in code)

Edit `app/rag_system.py` if you need to change:

- **Qdrant:** `QdrantClient("localhost", port=6333)`
- **Embeddings:** `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` (384 dimensions)
- **LLM:** `ChatOllama(model="llama3.2:1b", temperature=0)`
- **Chunking:** `chunk_size=450`, `chunk_overlap=120`
- **Retrieval:** `search_kwargs={"k": 5}` (top 5 chunks)

---

## License

Use as needed for your project.
