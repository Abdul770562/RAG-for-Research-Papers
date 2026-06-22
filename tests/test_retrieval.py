from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.vector_store_manager import VectorStoreManager
from retrieval.retriever import DocumentRetriever


def main():

    pdf_path = Path("data/pdfs/Attention_is_all_you_need.pdf")

    # =====================================================
    # Load PDF
    # =====================================================

    loader = DocumentLoader(pdf_path)
    documents = loader.load()

    # =====================================================
    # Chunk Documents
    # =====================================================

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    # =====================================================
    # Initialize Vector Store
    # =====================================================

    vector_store = VectorStoreManager()

    vector_store.create_or_load(chunks)

    retriever = vector_store.get_retriever()

    # =====================================================
    # Retrieval Layer
    # =====================================================

    document_retriever = DocumentRetriever(retriever)

    query = "What is multi-head attention?"

    results = document_retriever.retrieve(query)

    # =====================================================
    # Display Results
    # =====================================================

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(results["query"])

    print("\n" + "=" * 80)
    print("RETRIEVED DOCUMENTS")
    print("=" * 80)
    print(len(results["documents"]))

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in results["sources"]:
        print(source)

    print("\n" + "=" * 80)
    print("CONTEXT PREVIEW")
    print("=" * 80)

    print(results["context"][:2000])

    print("\n" + "=" * 80)
    print("INDIVIDUAL DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(results["documents"], start=1):

        print(f"\nResult {i}")

        print("-" * 60)

        print("Metadata")

        for key, value in doc.metadata.items():
            print(f"{key:<15}: {value}")

        print("\nPreview")

        print(doc.page_content[:300])

        print()


if __name__ == "__main__":
    main()