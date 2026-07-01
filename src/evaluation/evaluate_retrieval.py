from pathlib import Path
import json
from typing import Any

from src.vector_store.retrieve import retrieve


def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """
    Precision@K = number of relevant retrieved documents in top K / K.
    """
    top_k = retrieved_doc_ids[:k]

    if k == 0:
        return 0.0

    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)

    return relevant_count / k


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """
    Reciprocal rank = 1 / rank of the first relevant retrieved document.
    If no relevant document is retrieved, return 0.
    """
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1 / index

    return 0.0


def evaluate_query(query_item: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """
    Run retrieval for one query and calculate metrics.
    """
    query = query_item["query"]
    relevant_doc_ids = set(query_item["relevant_doc_ids"])

    retrieved_chunks = retrieve(query=query, top_k=top_k)

    retrieved_doc_ids = [
        item["metadata"]["doc_id"]
        for item in retrieved_chunks
    ]

    retrieved_titles = [
        item["metadata"].get("title")
        for item in retrieved_chunks
    ]

    p_at_k = precision_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=top_k,
    )

    rr = reciprocal_rank(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
    )

    return {
        "query": query,
        "relevant_doc_ids": list(relevant_doc_ids),
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieved_titles": retrieved_titles,
        f"precision@{top_k}": p_at_k,
        "reciprocal_rank": rr,
    }


def write_markdown_report(results: list[dict[str, Any]], output_path: Path, top_k: int = 5) -> None:
    """
    Save evaluation results as a readable Markdown report.
    """
    mean_precision = sum(r[f"precision@{top_k}"] for r in results) / len(results)
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results)

    lines = []

    lines.append("# Retrieval Evaluation Results")
    lines.append("")
    lines.append(f"Number of test queries: {len(results)}")
    lines.append(f"Top K: {top_k}")
    lines.append(f"Mean Precision@{top_k}: {mean_precision:.3f}")
    lines.append(f"MRR: {mrr:.3f}")
    lines.append("")
    lines.append("## Per-query Results")
    lines.append("")

    for i, result in enumerate(results, start=1):
        lines.append(f"### Query {i}")
        lines.append("")
        lines.append(f"**Query:** {result['query']}")
        lines.append("")
        lines.append(f"**Relevant doc IDs:** `{result['relevant_doc_ids']}`")
        lines.append("")
        lines.append(f"**Retrieved doc IDs:** `{result['retrieved_doc_ids']}`")
        lines.append("")
        lines.append(f"**Precision@{top_k}:** {result[f'precision@{top_k}']:.3f}")
        lines.append("")
        lines.append(f"**Reciprocal Rank:** {result['reciprocal_rank']:.3f}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    queries_path = project_root / "Data" / "evaluation_queries.json"
    output_json_path = project_root / "outputs" / "evaluation_results.json"
    output_md_path = project_root / "outputs" / "evaluation_results.md"

    top_k = 5

    with queries_path.open("r", encoding="utf-8") as f:
        evaluation_queries = json.load(f)

    results = []

    for query_item in evaluation_queries:
        result = evaluate_query(query_item, top_k=top_k)
        results.append(result)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    write_markdown_report(
        results=results,
        output_path=output_md_path,
        top_k=top_k,
    )

    mean_precision = sum(r[f"precision@{top_k}"] for r in results) / len(results)
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results)

    print(f"Saved JSON results to {output_json_path}")
    print(f"Saved Markdown report to {output_md_path}")
    print(f"Mean Precision@{top_k}: {mean_precision:.3f}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()