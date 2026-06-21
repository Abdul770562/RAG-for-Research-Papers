import re
import time
from statistics import mean
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import logger


class DocumentChunker:
    """
    Splits LangChain Documents into smaller chunks suitable
    for embeddings and retrieval.

    Responsibilities
    ----------------
    - Normalize extracted text
    - Split documents
    - Preserve metadata
    - Generate chunk identifiers
    - Filter invalid chunks
    - Log chunk statistics
    """

    def __init__(self) -> None:

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize whitespace while preserving document structure.
        """

        text = re.sub(r"\r\n?", "\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        start_time = time.perf_counter()

        logger.info(
            "Chunking %d document(s)...",
            len(documents),
        )

        cleaned_documents = [
            Document(
                page_content=self._clean_text(doc.page_content),
                metadata=doc.metadata.copy(),
            )
            for doc in documents
        ]

        chunks = self.text_splitter.split_documents(
            cleaned_documents
        )

        final_chunks: List[Document] = []

        page_chunk_counter = {}

        chunk_index = 0

        for chunk in chunks:

            if (
                len(chunk.page_content.strip())
                < settings.min_chunk_length
            ):
                continue

            page = chunk.metadata.get("page", 0)

            page_chunk_counter.setdefault(page, 0)

            page_chunk_counter[page] += 1

            chunk.metadata["chunk_index"] = chunk_index

            chunk.metadata["chunk_id"] = (
                f"p{page}_c{page_chunk_counter[page]}"
            )

            final_chunks.append(chunk)

            chunk_index += 1

        elapsed = time.perf_counter() - start_time

        chunk_lengths = [
            len(chunk.page_content)
            for chunk in final_chunks
        ]

        logger.info("Chunk Statistics")
        logger.info("-----------------------------")
        logger.info(
            "Documents          : %d",
            len(documents),
        )
        logger.info(
            "Chunks             : %d",
            len(final_chunks),
        )
        logger.info(
            "Average Length     : %.2f chars",
            mean(chunk_lengths),
        )
        logger.info(
            "Minimum Length     : %d chars",
            min(chunk_lengths),
        )
        logger.info(
            "Maximum Length     : %d chars",
            max(chunk_lengths),
        )
        logger.info(
            "Processing Time    : %.4f sec",
            elapsed,
        )

        return final_chunks