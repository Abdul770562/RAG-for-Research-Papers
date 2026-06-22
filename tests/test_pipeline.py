from pathlib import Path

from generation.llm_service import LLMService
from generation.prompt_builder import PromptBuilder
from ingestion.chunker import DocumentChunker
from ingestion.loader import DocumentLoader
from ingestion.vector_store_manager import VectorStoreManager
from pipeline.rag_pipeline import RAGPipeline
from retrieval.retriever import DocumentRetriever


def main():

    pdf_path = Path(
        "data/pdfs/Attention_is_all_you_need.pdf"
    )

    # ------------------------------------------------
    # Load
    # ------------------------------------------------

    loader = DocumentLoader(pdf_path)

    documents = loader.load()

    # ------------------------------------------------
    # Chunk
    # ------------------------------------------------

    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(documents)

    # ------------------------------------------------
    # Vector Store
    # ------------------------------------------------

    manager = VectorStoreManager()

    manager.create_or_load(chunks)

    retriever = manager.get_retriever()

    # ------------------------------------------------
    # Build Components
    # ------------------------------------------------

    document_retriever = DocumentRetriever(retriever)

    prompt_builder = PromptBuilder()

    llm = LLMService()

    pipeline = RAGPipeline(
        retriever=document_retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # ------------------------------------------------
    # Query
    # ------------------------------------------------

    query = (
        "Explain Multi-Head Attention."
    )

    result = pipeline.run(query)

    # ------------------------------------------------
    # Output
    # ------------------------------------------------

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(result["query"])

    print()

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])

    print()

    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()