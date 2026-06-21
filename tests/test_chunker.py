from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker


def main():
    pdf_path = Path("data/pdfs/Attention_is_all_you_need.pdf")

    # Load PDF
    loader = DocumentLoader(pdf_path)
    documents = loader.load()

    # Chunk Documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    print("=" * 80)
    print(f"Pages Loaded      : {len(documents)}")
    print(f"Chunks Generated  : {len(chunks)}")
    print("=" * 80)

    # Display first chunk
    first_chunk = chunks[0]

    print("\nFirst Chunk Metadata")
    print("-" * 80)

    for key, value in first_chunk.metadata.items():
        print(f"{key:<15}: {value}")

    print("\nChunk Length")
    print("-" * 80)
    print(len(first_chunk.page_content))

    print("\nFirst Chunk Preview")
    print("-" * 80)
    print(first_chunk.page_content[:500])

    print("\nLast Chunk Metadata")
    print("-" * 80)

    last_chunk = chunks[-1]

    for key, value in last_chunk.metadata.items():
        print(f"{key:<15}: {value}")

    print("\nLast Chunk Length")
    print("-" * 80)
    print(len(last_chunk.page_content))

    print("\nLast Chunk Preview")
    print("-" * 80)
    print(last_chunk.page_content[:500])


if __name__ == "__main__":
    main()