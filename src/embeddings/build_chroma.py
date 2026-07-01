from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path: Path) -> list:
    """
    Load chunks from chunks.json.
    """
    with chunks_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_chroma_vectorstore(
    chunks_path: Path,
    chroma_dir: Path,
    collection_name: str = "rag_documents",
    model_name: str = "all-MiniLM-L6-v2",
    reset_collection: bool = True,
):
    """
    Build a persistent Chroma vector database from text chunks.
    """
    chunks = load_chunks(chunks_path)

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)

    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        texts.append(chunk["text"])
        metadatas.append({
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source_html": chunk["source_html"],
        })

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    print("Adding embeddings to Chroma...")

    batch_size = 500

    for start in range(0, len(ids), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"Inserted {min(end, len(ids))}/{len(ids)} chunks.")

    print(f"Chroma vector store saved to {chroma_dir}")
    print(f"Collection name: {collection_name}")

    return collection


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    chunks_path = project_root / "Data" / "chunks.json"
    chroma_dir = project_root / "Data" / "chroma_db"

    build_chroma_vectorstore(
        chunks_path=chunks_path,
        chroma_dir=chroma_dir,
        collection_name="rag_documents",
        model_name="all-MiniLM-L6-v2",
        reset_collection=True,
    )