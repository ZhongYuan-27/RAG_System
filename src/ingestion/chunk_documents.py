from pathlib import Path
import json


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap.")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if len(chunk.strip()) > 100:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks(
    documents_path: Path,
    output_path: Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list:
    """
    Read documents.json and create chunks.json.
    """
    with documents_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            doc["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for i, chunk in enumerate(chunks, start=1):
            chunk_id = f"{doc['doc_id']}_chunk_{i:03d}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_html": doc["source_html"],
                "text": chunk,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return all_chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    documents_path = project_root / "Data" / "documents.json"
    output_path = project_root / "Data" / "chunks.json"

    chunks = build_chunks(
        documents_path=documents_path,
        output_path=output_path,
        chunk_size=500,
        overlap=100,
    )

    print(f"Saved {len(chunks)} chunks to {output_path}")