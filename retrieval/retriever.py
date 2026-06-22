# from typing import Dict, List

# from langchain_core.documents import Document
# from langchain_core.vectorstores import VectorStoreRetriever

# from utils.logger import logger


# class DocumentRetriever:
#     """
#     Handles document retrieval from the vector store.

#     Responsibilities
#     ----------------
#     - Retrieve relevant chunks
#     - Remove duplicate chunks
#     - Sort chunks
#     - Build context for the LLM
#     """

#     def __init__(
#         self,
#         retriever: VectorStoreRetriever,
#     ) -> None:

#         self.retriever = retriever

#     def retrieve(
#         self,
#         query: str,
#     ) -> Dict:

#         logger.info(
#             "Retrieving documents for query: %s",
#             query,
#         )

#         documents = self.retriever.invoke(query)

#         documents = self._remove_duplicates(documents)

#         documents = self._sort_documents(documents)

#         context = self._build_context(documents)

#         logger.info(
#             "Retrieved %d unique chunks.",
#             len(documents),
#         )

#         return {
#             "query": query,
#             "documents": documents,
#             "context": context,
#             "sources": self._extract_sources(documents),
#         }

#     @staticmethod
#     def _remove_duplicates(
#         documents: List[Document],
#     ) -> List[Document]:

#         seen = set()

#         unique_documents = []

#         for document in documents:

#             chunk_id = document.metadata["chunk_id"]

#             if chunk_id in seen:
#                 continue

#             seen.add(chunk_id)

#             unique_documents.append(document)

#         return unique_documents

#     @staticmethod
#     def _sort_documents(
#         documents: List[Document],
#     ) -> List[Document]:

#         return sorted(
#             documents,
#             key=lambda doc: (
#                 doc.metadata["page"],
#                 doc.metadata["chunk_index"],
#             ),
#         )

#     @staticmethod
#     def _build_context(
#         documents: List[Document],
#     ) -> str:

#         context = []

#         for document in documents:

#             page = document.metadata["page"]

#             chunk = document.metadata["chunk_id"]

#             context.append(
#                 f"[Page {page} | {chunk}]\n"
#                 f"{document.page_content}"
#             )

#         return "\n\n".join(context)

#     @staticmethod
#     def _extract_sources(
#         documents: List[Document],
#     ) -> List[Dict]:

#         sources = []

#         for document in documents:

#             sources.append(
#                 {
#                     "page": document.metadata["page"],
#                     "chunk_id": document.metadata["chunk_id"],
#                     "file_name": document.metadata["file_name"],
#                 }
#             )

#         return sources

from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from utils.logger import logger


class DocumentRetriever:
    """
    Handles retrieval of relevant document chunks from the vector store.

    Responsibilities
    ----------------
    - Retrieve relevant chunks
    - Remove duplicate chunks
    - Build formatted context
    - Extract source metadata
    """

    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Retrieve relevant chunks for a user query.
        """

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        logger.info("Retrieving documents for query: '%s'", query)

        documents = self.retriever.invoke(query)

        documents = self._remove_duplicates(documents)

        context = self._build_context(documents)

        logger.info(
            "Retrieved %d unique chunk(s).",
            len(documents),
        )

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
        """
        Remove duplicate chunks while preserving retrieval order.
        """

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
        """
        Build formatted context for the LLM.
        """

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
        """
        Extract source information from retrieved chunks.
        """

        sources = []

        seen = set()

        for document in documents:

            source = {
                "page": document.metadata["page"],
                "chunk_id": document.metadata["chunk_id"],
                "file_name": document.metadata["file_name"],
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