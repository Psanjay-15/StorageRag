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


