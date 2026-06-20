from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


class DocumentLoader:
    """
    Loads a research paper PDF into LangChain Document objects.

    Responsibilities:
    - Validate the PDF path.
    - Validate the file type.
    - Load the PDF using PyMuPDF.
    - Standardize metadata for downstream components.

    This class does NOT perform:
    - Chunking
    - Embedding generation
    - Vector database operations
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)

    def _validate_pdf(self) -> None:
        """
        Validate that the provided path exists and is a PDF.

        Raises:
            FileNotFoundError:
                If the PDF does not exist.

            ValueError:
                If the file is not a PDF.
        """

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        if not self.pdf_path.is_file():
            raise ValueError(
                f"Expected a file but received: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported file type '{self.pdf_path.suffix}'. "
                "Only PDF files are supported."
            )

    def load(self) -> List[Document]:
        """
        Load the PDF and standardize metadata.

        Returns:
            List[Document]:
                One Document object per page.

        Raises:
            RuntimeError:
                If PyMuPDF fails to load the document.
        """

        self._validate_pdf()

        try:
            loader = PyMuPDFLoader(str(self.pdf_path))
            documents = loader.load()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PDF: {self.pdf_path}"
            ) from exc

        standardized_documents = []

        for document in documents:

            metadata = document.metadata.copy()

            metadata["source"] = str(self.pdf_path.resolve())
            metadata["file_name"] = self.pdf_path.name
            metadata["page"] = metadata.get("page", 0)
            metadata["total_pages"] = metadata.get("total_pages")

            standardized_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return standardized_documents