import streamlit as st

from config import RETRIEVAL_TOP_K
from rag_pipeline import (
    build_vector_store,
    load_pdf,
    retrieve_context,
    split_documents,
)

st.set_page_config(page_title="Technical Document RAG Assistant", page_icon="📄")

st.title("Technical Document RAG Assistant")
st.write(
    "Upload a technical PDF, such as a CV, report, or engineering document, "
    "and retrieve the most relevant source sections for a question."
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    try:
        with st.spinner("Reading, chunking, and indexing PDF..."):
            documents = load_pdf(uploaded_file)
            chunks = split_documents(documents)
            vector_store = build_vector_store(chunks)

        st.success(
            f"Indexed {len(documents)} pages into {len(chunks)} searchable chunks."
        )

    except Exception as error:
        st.error("Something went wrong while processing the PDF.")
        st.caption(str(error))
        st.stop()

    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Retrieving relevant source sections..."):
            results = retrieve_context(
                vector_store,
                question,
                k=RETRIEVAL_TOP_K,
            )

        st.subheader("Retrieved source sections")

        for index, document in enumerate(results, start=1):
            page = document.metadata.get("page", "unknown")
            page_display = page + 1 if isinstance(page, int) else page

            with st.expander(f"Result {index} | Page {page_display}"):
                st.markdown(document.page_content)

else:
    st.info("Upload a PDF to begin.")
