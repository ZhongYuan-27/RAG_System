from pathlib import Path
import os
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

from src.vector_store.retrieve import retrieve


load_dotenv()


def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into a context string for the LLM.
    """
    context_parts = []

    for item in retrieved_chunks:
        metadata = item.get("metadata", {})
        title = metadata.get("title", "Unknown title")
        source = metadata.get("source_html", "Unknown source")
        text = item.get("text", "")

        context_parts.append(
            f"[Source {item['rank']}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """
    Build the RAG prompt sent to Claude.
    """
    return f"""
You are a retrieval-augmented generation assistant.

Answer the user's question using only the context provided below.

Rules:
1. Use only the provided context.
2. If the context does not contain enough information, say: "I do not know based on the provided documents."
3. Do not invent facts.
4. Cite the source title or source number when possible.
5. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def generate_answer(question: str, top_k: int = 5) -> dict[str, Any]:
    """
    Retrieve relevant chunks and generate an answer with Claude.
    """
    retrieved_chunks = retrieve(query=query_clean(question), top_k=top_k)

    context = format_context(retrieved_chunks)
    prompt = build_prompt(question, context)

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is missing. Please set it in the .env file.")

    api_key = api_key.strip()

    if not api_key.startswith("sk-ant-"):
        raise ValueError(
            "ANTHROPIC_API_KEY looks invalid. It should start with 'sk-ant-'. "
            "Check your .env file and remove placeholder text or non-English characters."
            )

    client = Anthropic(api_key=api_key)

    model_name = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    response = client.messages.create(
        model=model_name,
        max_tokens=800,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer_text = response.content[0].text

    sources = []

    usage = response.usage

    for item in retrieved_chunks:
        metadata = item.get("metadata", {})

        sources.append({
            "rank": item["rank"],
            "chunk_id": item["chunk_id"],
            "title": metadata.get("title"),
            "source_html": metadata.get("source_html"),
            "distance": item.get("distance"),
        })

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
    }


def query_clean(question: str) -> str:
    """
    Basic query cleaning.
    """
    return question.strip()


if __name__ == "__main__":
    test_question = "What is net present value?"

    result = generate_answer(test_question, top_k=5)

    print("Question:")
    print(result["question"])
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(source)