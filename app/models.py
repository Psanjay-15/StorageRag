from pydantic import BaseModel


class Query(BaseModel):
    question: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    collection_name: str
    chunk_count: int
