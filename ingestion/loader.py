from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from exceptions.document_exceptions import (
    DocumentLoadError,
    InvalidDocumentError,
)
from utils.logger import logger


class DocumentLoader:
    """
    Loads research paper PDFs into LangChain Documents.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)

    def _validate_pdf(self) -> None:
        """
        Validate the provided PDF.
        """

        if not self.pdf_path.exists():
            raise InvalidDocumentError(
                f"File does not exist: {self.pdf_path}"
            )

        if not self.pdf_path.is_file():
            raise InvalidDocumentError(
                f"Expected a file but got: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise InvalidDocumentError(
                "Only PDF documents are supported."
            )

    def _normalize_metadata(self, metadata: dict) -> dict:
        """
        Keep only metadata useful for retrieval and citations.
        """

        return {
            "source": str(self.pdf_path.resolve()),
            "file_name": self.pdf_path.name,
            "page": metadata.get("page", 0),
            "total_pages": metadata.get("total_pages"),
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
        }

    def load(self) -> List[Document]:
        """
        Load PDF pages as LangChain Documents.
        """

        self._validate_pdf()

        logger.info("Loading PDF: %s", self.pdf_path.name)

        try:
            loader = PyMuPDFLoader(str(self.pdf_path))
            documents = loader.load()

            standardized_documents = []

            for doc in documents:

                standardized_documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=self._normalize_metadata(doc.metadata),
                    )
                )

            logger.info(
                "Successfully loaded %d pages from %s",
                len(standardized_documents),
                self.pdf_path.name,
            )

            return standardized_documents

        except Exception as exc:
            logger.exception(
                "Failed to load document: %s",
                self.pdf_path.name,
            )

            raise DocumentLoadError(
                f"Unable to load PDF '{self.pdf_path.name}'."
            ) from exc