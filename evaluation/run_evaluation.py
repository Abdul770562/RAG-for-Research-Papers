import json
import time
from pathlib import Path

from generation.llm_service import LLMService
from generation.prompt_builder import PromptBuilder
from ingestion.chunker import DocumentChunker
from ingestion.loader import DocumentLoader
from ingestion.vector_store_manager import VectorStoreManager
from pipeline.rag_pipeline import RAGPipeline
from retrieval.retriever import DocumentRetriever
from utils.logger import logger


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "pdfs" / "Attention_is_all_you_need.pdf"

DATASET_PATH = (
    BASE_DIR
    / "evaluation"
    / "datasets"
    / "attention_is_all_you_need.json"
)

OUTPUT_DIR = BASE_DIR / "evaluation" / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "baseline_results.json"


# ==========================================================
# BUILD PIPELINE
# ==========================================================

def build_pipeline() -> RAGPipeline:
    """
    Build the complete RAG pipeline.
    """

    logger.info("Building RAG pipeline...")

    loader = DocumentLoader(PDF_PATH)
    documents = loader.load()

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    manager = VectorStoreManager()
    manager.create_or_load(chunks)

    retriever = manager.get_retriever()

    document_retriever = DocumentRetriever(retriever)

    prompt_builder = PromptBuilder()

    llm = LLMService()

    return RAGPipeline(
        retriever=document_retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset() -> list[dict]:
    """
    Load evaluation dataset.
    """

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(results: list[dict]) -> None:
    """
    Save evaluation results.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info("Saved results to %s", OUTPUT_FILE)


# ==========================================================
# MAIN
# ==========================================================

def main():

    dataset = load_dataset()

    pipeline = build_pipeline()

    results = []

    logger.info("Running evaluation on %d questions...", len(dataset))

    total_start = time.perf_counter()

    for idx, sample in enumerate(dataset, start=1):

        question = sample["question"]

        logger.info(
            "[%d/%d] %s",
            idx,
            len(dataset),
            question,
        )

        start = time.perf_counter()

        output = pipeline.run(question)

        elapsed = time.perf_counter() - start

        results.append(
            {
                "id": sample["id"],
                "question": question,
                "category": sample.get("category"),
                "difficulty": sample.get("difficulty"),
                "answer": output["answer"],
                "contexts": [
                    doc.page_content
                    for doc in output["documents"]
                ],
                "sources": output["sources"],
                "latency_seconds": round(elapsed, 3),
            }
        )

    total_elapsed = time.perf_counter() - total_start

    save_results(results)

    logger.info(
        "Evaluation completed in %.2f seconds.",
        total_elapsed,
    )


if __name__ == "__main__":
    main()