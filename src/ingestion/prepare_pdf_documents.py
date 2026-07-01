from pathlib import Path
import json
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages_text.append(text)

    return "\n\n".join(pages_text), len(reader.pages)


def clean_doc_id(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(",", "")
        .replace("__", "_")
    )


def prepare_pdf_documents(
    pdf_root_dir: Path,
    output_path: Path,
    min_length: int = 300,
) -> list[dict]:
    documents = []

    pdf_files = sorted(pdf_root_dir.rglob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.relative_to(pdf_root_dir)}...")

        content, page_count = extract_text_from_pdf(pdf_path)

        if len(content) < min_length:
            print(f"Skipped {pdf_path.name}: extracted text too short.")
            continue

        category = pdf_path.parent.name
        file_stem = pdf_path.stem

        doc_id = clean_doc_id(f"{category}_{file_stem}")

        title = f"{category} - {file_stem}"

        document = {
            "doc_id": doc_id,
            "title": title,
            "source": str(pdf_path),
            "source_path": str(pdf_path),
            "source_html": str(pdf_path),  # keep this for compatibility with existing code
            "content": content,
            "metadata": {
                "file_type": "pdf",
                "category": category,
                "page_count": page_count,
                "original_filename": pdf_path.name,
                "relative_path": str(pdf_path.relative_to(pdf_root_dir)),
            },
        }

        documents.append(document)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    return documents


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    pdf_root_dir = project_root / "Data" / "raw" / "pdfs"
    output_path = project_root / "Data" / "documents.json"

    docs = prepare_pdf_documents(
        pdf_root_dir=pdf_root_dir,
        output_path=output_path,
    )

    print(f"Saved {len(docs)} documents to {output_path}")
