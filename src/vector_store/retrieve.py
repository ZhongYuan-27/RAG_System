from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


class ChromaRetriever:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str = "rag_documents",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(name=collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved = []

        for i in range(len(results["ids"][0])):
            item = {
                "rank": i + 1,
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }

            retrieved.append(item)

        return retrieved


def get_default_retriever() -> ChromaRetriever:
    project_root = Path(__file__).resolve().parents[2]
    chroma_dir = project_root / "Data" / "chroma_db"

    return ChromaRetriever(
        chroma_dir=chroma_dir,
        collection_name="rag_documents",
        model_name="all-MiniLM-L6-v2",
    )


def retrieve(query: str, top_k: int = 5) -> list:
    retriever = get_default_retriever()
    return retriever.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    retriever = get_default_retriever()

    test_query = "What is net present value?"

    results = retriever.retrieve(test_query, top_k=5)

    print(f"Query: {test_query}")
    print("=" * 80)

    for item in results:
        print(f"Rank: {item['rank']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Title: {item['metadata'].get('title')}")
        print(f"Source: {item['metadata'].get('source_html')}")
        print(f"Distance: {item['distance']}")
        print(item["text"][:800])
        print("=" * 80)