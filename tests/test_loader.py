from pathlib import Path

from ingestion.loader import DocumentLoader


def main():

    pdf = Path("data\pdfs\Attention_is_all_you_need.pdf")

    loader = DocumentLoader(pdf)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    print()

    print(documents[0].metadata)

    print()

    print(documents[0].page_content[:500])


if __name__ == "__main__":
    main()