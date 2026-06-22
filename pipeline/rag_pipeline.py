from typing import Any

from generation.llm_service import LLMService
from generation.prompt_builder import PromptBuilder
from retrieval.retriever import DocumentRetriever


class RAGPipeline:
    """
    Coordinates the Retrieval-Augmented Generation pipeline.

    Flow
    ----
    User Query
        ↓
    Retrieval
        ↓
    Prompt Builder
        ↓
    LLM
        ↓
    Final Answer
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        prompt_builder: PromptBuilder,
        llm: LLMService,
    ) -> None:

        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm

    def run(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Execute the complete RAG pipeline.
        """

        retrieval_result = self.retriever.retrieve(query)

        messages = self.prompt_builder.build(retrieval_result)

        answer = self.llm.generate(messages)

        return {
            "query": query,
            "answer": answer,
            "sources": retrieval_result["sources"],
            "documents": retrieval_result["documents"],
        }