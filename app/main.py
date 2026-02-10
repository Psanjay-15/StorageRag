from fastapi import FastAPI, UploadFile, File, HTTPException
from .models import Query, UploadResponse
from .rag_system import index_pdf_from_bytes, generate_answer
import traceback

app = FastAPI(title="StorageRag")


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_and_index_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only .pdf files are allowed")

    try:
        pdf_bytes = await file.read()

        if len(pdf_bytes) < 100:
            raise ValueError("File is empty or too small to be a valid PDF")

        # Index directly from bytes — NO disk write
        result = index_pdf_from_bytes(pdf_bytes, file.filename)

        return UploadResponse(
            message="PDF processed and indexed in Qdrant (no file saved to disk)",
            filename=file.filename,
            collection_name=result["collection_name"],
            chunk_count=result["chunk_count"],
        )

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        print(f"Upload failed:\n{tb}")
        raise HTTPException(500, detail=f"Processing failed: {str(e)}")


@app.post("/query")
async def ask_question(query: Query):
    try:
        answer = generate_answer(query.question)
        return {"answer": answer}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"Query failed: {str(e)}")


@app.get("/health")
async def health():
    from .rag_system import _active_collection, _active_retriever

    return {
        "status": "ok",
        "active_collection": _active_collection,
        "has_active_index": _active_retriever is not None,
    }
