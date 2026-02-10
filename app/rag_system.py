import uuid
from typing import Optional
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .utils.pdf_processor import extract_text_from_pdf_bytes

# Globals
client = QdrantClient("localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3.2:1b", temperature=0)

_active_vectorstore: Optional[QdrantVectorStore] = None
_active_retriever = None
_active_collection: Optional[str] = None


SYSTEM_PROMPT = """\
You are a helpful enterprise assistant that answers questions using **only** the provided document context.
If the information is not present in the context, reply exactly:
"The information is not available in the provided company documents."

Context:
{context}
"""


def get_safe_collection_name(original_filename: str) -> str:
    safe_base = (
        original_filename.lower()
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")[:32]
    )
    short_uuid = str(uuid.uuid4())[:8]
    return f"pdf_{safe_base}_{short_uuid}"


def index_pdf_from_bytes(pdf_bytes: bytes, original_filename: str) -> dict:
    """
    Parse → chunk → embed → store in Qdrant
    No file is saved to disk
    """
    global _active_vectorstore, _active_retriever, _active_collection

    full_text = extract_text_from_pdf_bytes(pdf_bytes)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "!", "?", " ", ""],
    )

    chunks = splitter.split_text(full_text)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": original_filename,
                "file_id": str(uuid.uuid4())[:12],
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    collection_name = get_safe_collection_name(original_filename)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    vector_store.add_documents(documents)

    _active_vectorstore = vector_store
    _active_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    _active_collection = collection_name

    print(f"Indexed {len(chunks)} chunks into collection: {collection_name}")

    return {"collection_name": collection_name, "chunk_count": len(chunks)}


def get_current_retriever():
    if _active_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="No document is currently indexed. Please upload a PDF first.",
        )
    return _active_retriever


def generate_answer(question: str) -> str:
    retriever = get_current_retriever()
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant content found in the document."

    context = "\n\n".join(d.page_content for d in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
