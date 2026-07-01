# RAG System for PDF Question Answering

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions based on PDF documents about corporate finance.

The system processes raw PDF files, extracts text, splits documents into smaller chunks, builds a vector database using embeddings, retrieves relevant document chunks for a user query, and generates answers with a large language model.

## Project Overview

The main goal of this project is to build an end-to-end RAG pipeline that can:

1. Load and process PDF documents.
2. Convert documents into searchable text chunks.
3. Store document embeddings in a Chroma vector database.
4. Retrieve relevant chunks based on a user query.
5. Generate grounded answers using an LLM.
6. Evaluate retrieval performance using predefined evaluation queries.

## Project Structure

```text
RAG_System/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── Data/
│   ├── raw/
│   ├── chroma_db/
│   ├── documents.json
│   ├── chunks.json
│   └── evaluation_queries.json
│
├── notebooks/
│   ├── prepare_pdf_documents.ipynb
│   ├── chunk_documents.ipynb
│   ├── embeddings_and_vectorstore.ipynb
│   └── retrieval_test.ipynb
│
├── outputs/
│   ├── evaluation_results.json
│   └── evaluation_results.md
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── prepare_pdf_documents.py
│   │   └── chunk_documents.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── build_chroma.py
│   │
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── retrieve.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   └── answer.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_retrieval.py
│   │
│   ├── __init__.py
│   └── split_pdf.py
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── test_claude.py
```

## Main Components

### 1. PDF Ingestion

The ingestion module reads PDF files from the raw data folder and extracts their text content.

Relevant files:

```text
src/ingestion/prepare_pdf_documents.py
notebooks/prepare_pdf_documents.ipynb
```

Output:

```text
Data/documents.json
```

### 2. Document Chunking

The chunking module splits extracted documents into smaller text chunks so that they can be embedded and retrieved efficiently.

Relevant files:

```text
src/ingestion/chunk_documents.py
notebooks/chunk_documents.ipynb
```

Output:

```text
Data/chunks.json
```

### 3. Embedding and Vector Database

The embedding module converts document chunks into vector embeddings and stores them in a Chroma vector database.

Relevant files:

```text
src/embeddings/build_chroma.py
notebooks/embeddings_and_vectorstore.ipynb
```

Output:

```text
Data/chroma_db/
```

### 4. Retrieval

The retrieval module searches the vector database and returns the most relevant chunks for a given user query.

Relevant file:

```text
src/vector_store/retrieve.py
```

### 5. RAG Answer Generation

The RAG module combines retrieved document chunks with a user query and sends them to a large language model to generate a grounded answer.

Relevant file:

```text
src/rag/answer.py
```

### 6. Evaluation

The evaluation module tests whether the retriever can return relevant chunks for predefined evaluation queries.

Relevant files:

```text
src/evaluation/evaluate_retrieval.py
Data/evaluation_queries.json
```

Outputs:

```text
outputs/evaluation_results.json
outputs/evaluation_results.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ZhongYuan-27/RAG_System.git
cd RAG_System
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file based on `.env.example`.

```bash
copy .env.example .env
```

Then add your API key inside `.env`.

Example:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

or, if using OpenAI:

```env
OPENAI_API_KEY=your_api_key_here
```

The `.env` file should not be committed to GitHub.

## Data Preparation

Place your PDF files in:

```text
Data/raw/
```

If the project uses a specific PDF folder, place them in:

```text
Data/raw/pdfs/
```

Raw PDF files, generated chunks, and vector databases are usually excluded from GitHub to avoid uploading large or private files.

## Usage

Run the following commands from the project root directory.

### 1. Extract text from PDF documents

```bash
python -m src.ingestion.prepare_pdf_documents
```

This generates:

```text
Data/documents.json
```

### 2. Split documents into chunks

```bash
python -m src.ingestion.chunk_documents
```

This generates:

```text
Data/chunks.json
```

### 3. Build the Chroma vector database

```bash
python -m src.embeddings.build_chroma
```

This generates:

```text
Data/chroma_db/
```

### 4. Test retrieval

```bash
python -m src.vector_store.retrieve
```

Or use the notebook:

```text
notebooks/retrieval_test.ipynb
```

### 5. Generate RAG answers

If the answer generation module is configured, run:

```bash
python -m src.rag.answer
```

## Running the API

This project includes a FastAPI application.

Start the API server with:

```bash
uvicorn app.main:app --reload
```

Then open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

You can use the API endpoint to submit a question and receive an answer based on the indexed documents.

## Evaluation

To evaluate retrieval performance, run:

```bash
python -m src.evaluation.evaluate_retrieval
```

The evaluation queries are stored in:

```text
Data/evaluation_queries.json
```

Evaluation results are saved to:

```text
outputs/evaluation_results.json
outputs/evaluation_results.md
```

## Notebooks

The `notebooks/` folder contains step-by-step demonstrations of the pipeline:

```text
prepare_pdf_documents.ipynb
chunk_documents.ipynb
embeddings_and_vectorstore.ipynb
retrieval_test.ipynb
```

These notebooks are mainly used for experimentation, debugging, and project demonstration.

The main reusable code is stored in the `src/` folder.

## Files Not Included in GitHub

The following files or folders are usually excluded from version control:

```text
.venv/
.env
__pycache__/
Data/raw/
Data/chroma_db/
Data/documents.json
Data/chunks.json
outputs/
```

This keeps the repository clean and avoids uploading private data, API keys, generated files, or large vector databases.

## Example Workflow

A typical workflow is:

```text
Raw PDFs
   ↓
Extract text
   ↓
documents.json
   ↓
Chunk documents
   ↓
chunks.json
   ↓
Generate embeddings
   ↓
Chroma vector database
   ↓
Retrieve relevant chunks
   ↓
Generate answer with LLM
```

## Tech Stack

- Python
- FastAPI
- ChromaDB
- Large Language Models
- PDF text processing
- Vector embeddings
- Jupyter Notebook
- Git and GitHub

## Future Improvements

Possible future improvements include:

- Add a web-based user interface.
- Improve chunking strategy.
- Add source citations in generated answers.
- Support multiple document formats such as TXT, DOCX, and HTML.
- Add more retrieval evaluation metrics.
- Improve prompt design for answer generation.
- Add Docker support for easier deployment.

## License

This project is for educational and portfolio purposes.

## Author

ZhongYuan-27
