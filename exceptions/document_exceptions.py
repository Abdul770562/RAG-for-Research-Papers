"""
Custom exceptions for document ingestion.
"""


class DocumentError(Exception):
    """Base exception for all document-related errors."""


class InvalidDocumentError(DocumentError):
    """Raised when an invalid document is provided."""


class DocumentLoadError(DocumentError):
    """Raised when a document cannot be loaded."""