from pathlib import Path
from pypdf import PdfReader, PdfWriter


INPUT_DIR = Path("data/raw/original_pdfs")
OUTPUT_DIR = Path("data/raw/pdfs")
NUM_PARTS = 60


def split_pdf_evenly(input_pdf: Path, output_base_dir: Path, num_parts: int = 30):
    reader = PdfReader(str(input_pdf))
    total_pages = len(reader.pages)

    base_pages = total_pages // num_parts
    extra_pages = total_pages % num_parts

    pdf_output_dir = output_base_dir / input_pdf.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    current_page = 0

    for i in range(num_parts):
        pages_in_this_part = base_pages + (1 if i < extra_pages else 0)

        if pages_in_this_part == 0:
            break

        writer = PdfWriter()

        start_page = current_page
        end_page = current_page + pages_in_this_part

        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        output_path = pdf_output_dir / f"{input_pdf.stem}_part_{i + 1:02d}.pdf"

        with open(output_path, "wb") as f:
            writer.write(f)

        print(f"{input_pdf.name}: created part {i + 1:02d}, pages {start_page + 1}-{end_page}")

        current_page = end_page


def main():
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    for pdf_file in pdf_files:
        split_pdf_evenly(pdf_file, OUTPUT_DIR, NUM_PARTS)

    print("All PDFs have been split.")


if __name__ == "__main__":
    main()