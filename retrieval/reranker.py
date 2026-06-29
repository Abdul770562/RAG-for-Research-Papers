from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config.settings import settings
from utils.logger import logger


class DocumentReranker:
    """
    Cross-Encoder based document reranker.

    Retrieves relevance scores for (query, document)
    pairs and sorts documents accordingly.
    """

    def __init__(self) -> None:

        logger.info(
            "Loading reranker model: %s",
            settings.reranker_model,
        )

        self.model = CrossEncoder(
            settings.reranker_model,
            trust_remote_code=True,
        )

        logger.info("Cross Encoder loaded.")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[Document]:

        if not documents:
            return []

        if top_k is None:
            top_k = settings.final_top_k

        pairs = [
            (
                query,
                doc.page_content,
            )
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked_documents = []

        for document, score in ranked[:top_k]:

            document.metadata["rerank_score"] = float(score)

            reranked_documents.append(document)

        return reranked_documents