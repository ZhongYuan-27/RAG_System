from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.answer import generate_answer


app = FastAPI(
    title="RAG System API",
    description="A small Retrieval-Augmented Generation system using semantic retrieval and Claude.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="User question to ask the RAG system.",
        examples=["What is the German credit dataset about?"],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of retrieved chunks to use as context.",
    )


class Source(BaseModel):
    rank: int
    chunk_id: str
    title: str | None = None
    source_html: str | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/")
def root() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "message": "RAG System API is running.",
        "docs": "Visit /docs to test the API.",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, Any]:
    """
    Ask a question to the RAG system.
    """
    try:
        result = generate_answer(
            question=request.question,
            top_k=request.top_k,
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )