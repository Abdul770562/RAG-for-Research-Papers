from huggingface_hub.inference._generated.types import zero_shot_image_classification
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from retrieval.reranker import DocumentReranker
from utils.logger import logger


class DocumentRetriever:
    """
    Handles retrieval of relevant document chunks.

    Responsibilities
    ----------------
    - Dense retrieval from vector store
    - Duplicate removal
    - Cross-encoder reranking (optional)
    - Context construction
    - Source extraction
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        reranker: DocumentReranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Retrieve relevant chunks for a user query.
        """

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        logger.info(
            "Retrieving top %d candidates for query: '%s'",
            settings.retrieval_top_k,
            query,
        )

        documents = self.retriever.invoke(query)

        documents = self._remove_duplicates(documents)

        logger.info(
            "Retrieved %d unique candidate chunk(s).",
            len(documents),
        )

        if self.reranker is not None:

            logger.info("Applying cross-encoder reranking...")

            documents = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=settings.final_top_k,
            )

            logger.info(
                "Selected top %d reranked chunk(s).",
                len(documents),
            )

        context = self._build_context(documents)

        return {
            "query": query,
            "documents": documents,
            "context": context,
            "sources": self._extract_sources(documents),
        }

    @staticmethod
    def _remove_duplicates(
        documents: list[Document],
    ) -> list[Document]:

        seen = set()
        unique_documents = []

        for document in documents:

            chunk_id = document.metadata.get("chunk_id")

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            unique_documents.append(document)

        return unique_documents

    @staticmethod
    def _build_context(
        documents: list[Document],
    ) -> str:

        sections = []

        for document in documents:

            page = document.metadata["page"]
            chunk = document.metadata["chunk_id"]

            sections.append(
                "\n".join(
                    [
                        "=" * 60,
                        f"PAGE {page} | CHUNK {chunk}",
                        "=" * 60,
                        document.page_content.strip(),
                    ]
                )
            )

        return "\n\n".join(sections)

    @staticmethod
    def _extract_sources(
        documents: list[Document],
    ) -> list[dict[str, Any]]:

        sources = []
        seen = set()

        for document in documents:

            source = {
                "page": document.metadata["page"],
                "chunk_id": document.metadata["chunk_id"],
                "file_name": document.metadata["file_name"],
                "rerank_score": document.metadata.get("rerank_score"),
            }

            key = (
                source["page"],
                source["chunk_id"],
            )

            if key in seen:
                continue

            seen.add(key)
            sources.append(source)

        return sources