from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.vector_store_manager import VectorStoreManager


def main():

    pdf_path = Path(
        "data/pdfs/Attention_is_all_you_need.pdf"
    )

    # ----------------------------------
    # Load
    # ----------------------------------

    loader = DocumentLoader(pdf_path)

    documents = loader.load()

    # ----------------------------------
    # Chunk
    # ----------------------------------

    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(documents)

    # ----------------------------------
    # Vector Store
    # ----------------------------------

    manager = VectorStoreManager()

    manager.create_or_load(chunks)

    retriever = manager.get_retriever()

    print("=" * 80)
    print("VECTOR STORE CREATED")
    print("=" * 80)

    print(f"Chunks Indexed : {len(chunks)}")

    print(f"Retriever Type : {type(retriever).__name__}")

    print()

    # ----------------------------------
    # Test Retrieval
    # ----------------------------------

    query = "What is multi-head attention?"

    results = retriever.invoke(query)

    print("=" * 80)
    print(f"Query : {query}")
    print("=" * 80)

    print(f"Retrieved {len(results)} chunks\n")

    for idx, doc in enumerate(results, start=1):

        print("-" * 80)

        print(f"Result #{idx}")

        print()

        print("Metadata:")

        for key, value in doc.metadata.items():

            print(f"{key:<15}: {value}")

        print()

        print("Content Preview")

        print(doc.page_content[:400])

        print()


if __name__ == "__main__":
    main()