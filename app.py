# import time
# from pathlib import Path

# import streamlit as st

# from generation.llm_service import LLMService
# from generation.prompt_builder import PromptBuilder
# from ingestion.chunker import DocumentChunker
# from ingestion.loader import DocumentLoader
# from ingestion.vector_store_manager import VectorStoreManager
# from pipeline.rag_pipeline import RAGPipeline
# from retrieval.retriever import DocumentRetriever

# # ==========================================================
# # PAGE CONFIG
# # ==========================================================

# st.set_page_config(
#     page_title="Research Paper RAG",
#     page_icon="📄",
#     layout="wide",
# )

# st.title("📄 Research Paper RAG")

# st.caption("Production RAG Pipeline Demo")

# # ==========================================================
# # LOAD PIPELINE (CACHE)
# # ==========================================================


# @st.cache_resource(show_spinner="Loading RAG pipeline...")
# def load_pipeline():

#     pdf_path = Path(
#         "data/pdfs/Attention_is_all_you_need.pdf"
#     )

#     loader = DocumentLoader(pdf_path)

#     documents = loader.load()

#     chunker = DocumentChunker()

#     chunks = chunker.chunk_documents(documents)

#     manager = VectorStoreManager()

#     manager.create_or_load(chunks)

#     retriever = manager.get_retriever()

#     document_retriever = DocumentRetriever(retriever)

#     prompt_builder = PromptBuilder()

#     llm = LLMService()

#     pipeline = RAGPipeline(
#         retriever=document_retriever,
#         prompt_builder=prompt_builder,
#         llm=llm,
#     )

#     return pipeline, len(chunks), pdf_path.name


# pipeline, chunk_count, pdf_name = load_pipeline()

# # ==========================================================
# # SIDEBAR
# # ==========================================================

# with st.sidebar:

#     st.header("System Information")

#     st.write(f"**Paper:** {pdf_name}")

#     st.write(f"**Chunks:** {chunk_count}")

#     st.write("**Embedding:** BAAI/bge-small-en-v1.5")

#     st.write("**LLM:** llama-3.3-70b-versatile")

#     st.write("**Retriever:** ChromaDB")

# # ==========================================================
# # QUERY
# # ==========================================================

# query = st.text_input(
#     "Ask a question about the research paper"
# )

# # ==========================================================
# # RUN PIPELINE
# # ==========================================================

# if st.button("Generate Answer"):

#     if not query.strip():

#         st.warning("Please enter a question.")

#     else:

#         start = time.perf_counter()

#         result = pipeline.run(query)

#         elapsed = time.perf_counter() - start

#         st.divider()

#         st.subheader("Answer")

#         st.markdown(result["answer"])

#         st.divider()

#         st.subheader("Sources")

#         for source in result["sources"]:

#             st.markdown(
#                 f"""
# - **Page:** {source["page"]}
# - **Chunk:** {source["chunk_id"]}
# """
#             )

#         st.divider()

#         st.subheader("Retrieved Context")

#         for i, document in enumerate(result["documents"], start=1):

#             with st.expander(
#                 f"Chunk {i} | Page {document.metadata['page']} | {document.metadata['chunk_id']}"
#             ):

#                 st.text(document.page_content)

#         st.divider()

#         st.success(f"Completed in {elapsed:.2f} seconds")


import time
from pathlib import Path

import streamlit as st

from generation.llm_service import LLMService
from generation.prompt_builder import PromptBuilder
from ingestion.chunker import DocumentChunker
from ingestion.loader import DocumentLoader
from ingestion.vector_store_manager import VectorStoreManager
from pipeline.rag_pipeline import RAGPipeline
from retrieval.reranker import DocumentReranker
from retrieval.retriever import DocumentRetriever

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Research Paper RAG")

st.caption("Production RAG Pipeline Demo")

# ==========================================================
# LOAD PIPELINE (CACHE)
# ==========================================================


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline():

    pdf_path = Path(
        "data/pdfs/Attention_is_all_you_need.pdf"
    )

    loader = DocumentLoader(pdf_path)

    documents = loader.load()

    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(documents)

    manager = VectorStoreManager()

    manager.create_or_load(chunks)

    retriever = manager.get_retriever()

    # ------------------------------------------------------
    # Initialize Cross-Encoder Reranker
    # ------------------------------------------------------

    reranker = DocumentReranker()

    document_retriever = DocumentRetriever(
        retriever=retriever,
        reranker=reranker,
    )

    prompt_builder = PromptBuilder()

    llm = LLMService()

    pipeline = RAGPipeline(
        retriever=document_retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    return pipeline, len(chunks), pdf_path.name


pipeline, chunk_count, pdf_name = load_pipeline()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("System Information")

    st.write(f"**Paper:** {pdf_name}")

    st.write(f"**Chunks:** {chunk_count}")

    st.write("**Embedding:** BAAI/bge-small-en-v1.5")

    st.write("**Reranker:** BAAI/bge-reranker-base")

    st.write("**LLM:** llama-3.3-70b-versatile")

    st.write("**Retriever:** ChromaDB")

# ==========================================================
# QUERY
# ==========================================================

query = st.text_input(
    "Ask a question about the research paper"
)

# ==========================================================
# RUN PIPELINE
# ==========================================================

if st.button("Generate Answer"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        start = time.perf_counter()

        result = pipeline.run(query)

        elapsed = time.perf_counter() - start

        st.divider()

        st.subheader("Answer")

        st.markdown(result["answer"])

        st.divider()

        st.subheader("Sources")

        for source in result["sources"]:

            st.markdown(
                f"""
- **Page:** {source["page"]}
- **Chunk:** {source["chunk_id"]}
"""
            )

        st.divider()

        st.subheader("Retrieved Context")

        for i, document in enumerate(
            result["documents"],
            start=1,
        ):

            title = (
                f"Chunk {i} | "
                f"Page {document.metadata['page']} | "
                f"{document.metadata['chunk_id']}"
            )

            if "rerank_score" in document.metadata:
                title += (
                    f" | Score: "
                    f"{document.metadata['rerank_score']:.4f}"
                )

            with st.expander(title):

                st.text(document.page_content)

        st.divider()

        st.success(
            f"Completed in {elapsed:.2f} seconds"
        )