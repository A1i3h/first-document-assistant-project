import os
import re
import tempfile
from typing import List

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Clean extracted PDF text before chunking."""
    text = re.sub(r"\s+", " ", text)

    section_headings = [
        "PROFESSIONAL SUMMARY",
        "SKILLS",
        "EDUCATION",
        "WORK EXPERIENCE",
        "PROJECTS",
        "LANGUAGES",
    ]

    for heading in section_headings:
        text = text.replace(f" {heading}", f"\n\n{heading}")

    return text.strip()


def load_pdf(uploaded_file) -> List[Document]:
    """Load an uploaded PDF into LangChain document objects."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
    finally:
        os.remove(temp_path)

    for document in documents:
        document.page_content = clean_text(document.page_content)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split extracted document pages into retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document]):
    """Create a FAISS vector store using local sentence-transformer embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vector_store, question: str, k: int) -> List[Document]:
    """Retrieve the most relevant chunks for a question."""
    return vector_store.similarity_search(question, k=k)
