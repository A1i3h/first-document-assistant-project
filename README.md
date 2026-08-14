 Technical Document RAG Assistant

A Retrieval-Augmented Generation (RAG) app that lets you upload a technical PDF (a CV, report, or engineering document) and retrieve the most relevant source sections for a question, with page-level citations. Built with Python, Streamlit, LangChain, and FAISS.

## How it works

1. **Upload** a PDF through the Streamlit UI.
2. **Load & clean** — the PDF is parsed with `PyPDFLoader`, and extracted text is normalised (whitespace collapsed, common resume/report section headings re-broken onto their own lines) to reduce noise before chunking.
3. **Chunk** — cleaned pages are split into overlapping chunks using `RecursiveCharacterTextSplitter`, so relevant context isn't cut off mid-thought.
4. **Embed & index** — chunks are embedded locally with a `sentence-transformers` model and indexed in an in-memory **FAISS** vector store (no external embedding API calls).
5. **Retrieve** — your question is embedded and matched against the index via similarity search, returning the top-k most relevant chunks.
6. **Inspect** — each result is shown in an expandable card labelled with its source page number, so you can verify exactly where an answer came from.

This project focuses on the **retrieval** side of RAG: clean ingestion, sensible chunking, and explainable, source-grounded results, rather than passing retrieved context to an LLM for generation.

## Project structure

```
.
├── app/
│   ├── main.py            # Streamlit UI and app flow
│   ├── rag_pipeline.py    # PDF loading, cleaning, chunking, embedding, retrieval
│   └── config.py          # Chunk size/overlap, top-k, embedding model
├── data/                  # Local scratch space for uploaded/sample PDFs (gitignored)
├── tests/
│   └── example_questions.md
├── requirements.txt
└── .gitignore
```

## Configuration

Retrieval behaviour is controlled in `app/config.py`:

| Setting | Value | Description |
|---|---|---|
| `CHUNK_SIZE` | 550 | Max characters per chunk |
| `CHUNK_OVERLAP` | 100 | Character overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | 4 | Number of chunks retrieved per query |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model used for indexing and search |

## Getting started

**Prerequisites:** Python 3.9+

```bash
# Clone the repo
git clone https://github.com/A1i3h/first-document-assistant-project.git
cd first-document-assistant-project

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

The app opens in your browser (usually at `http://localhost:8501`). Upload a PDF, wait for it to be indexed, and start asking questions about it.

## Tech stack

- **Streamlit** — UI
- **LangChain** (`langchain`, `langchain-community`, `langchain-text-splitters`) — document loading, splitting, and vector store integration
- **FAISS** (`faiss-cpu`) — local vector similarity search
- **sentence-transformers** — local embedding generation
- **pypdf** — PDF parsing
- **pytest** — testing

## Possible next steps

- Add an LLM generation step on top of retrieval for direct, cited answers rather than raw source chunks
- Support multi-file uploads and cross-document search
- Persist the FAISS index to disk so re-uploading isn't required between sessions
- Expand automated test coverage beyond example questions

## License

No license specified.
