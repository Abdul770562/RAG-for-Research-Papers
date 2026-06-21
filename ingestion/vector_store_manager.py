from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from utils.logger import logger


class VectorStoreManager:
    """
    Manages ChromaDB operations.

    Responsibilities
    ----------------
    - Create a persistent Chroma database
    - Load an existing database
    - Index documents
    - Return a retriever
    """

    def __init__(self) -> None:

        self.persist_directory = Path(
            settings.persist_directory
        )

        self.collection_name = settings.collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        self.vector_store: Chroma | None = None

    def create_or_load(
        self,
        documents: List[Document],
    ) -> None:
        """
        Create a new vector database if it doesn't exist,
        otherwise load the existing database.
        """

        if self._collection_exists():

            logger.info(
                "Loading existing Chroma collection '%s'",
                self.collection_name,
            )

            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(
                    self.persist_directory
                ),
                embedding_function=self.embeddings,
            )

            return

        logger.info(
            "Creating new Chroma collection '%s'",
            self.collection_name,
        )

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(
                self.persist_directory
            ),
        )

        logger.info(
            "Indexed %d chunks.",
            len(documents),
        )

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Return a retriever.
        """

        if self.vector_store is None:

            raise RuntimeError(
                "Vector store has not been initialized."
            )

        return self.vector_store.as_retriever(
            search_kwargs={
                "k": settings.top_k
            }
        )

    def delete_collection(self) -> None:
        """
        Delete the Chroma collection.
        """

        if self.vector_store is not None:

            self.vector_store.delete_collection()

            logger.info(
                "Deleted collection '%s'",
                self.collection_name,
            )

    def _collection_exists(self) -> bool:
        """
        Determine whether a persisted Chroma database exists.
        """

        return (
            self.persist_directory.exists()
            and any(self.persist_directory.iterdir())
        )