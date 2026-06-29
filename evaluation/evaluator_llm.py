# """
# evaluation/evaluator_llm.py

# Creates the modern InstructorLLM evaluator used natively by RAGAS collections.
# Routes custom open-source models through the Groq hardware acceleration cloud gateway.
# """
# from langchain_huggingface import HuggingFaceEmbeddings

# from ragas.embeddings import HuggingFaceEmbeddings
# from ragas.llms import llm_factory

# from config.settings import settings
# from utils.logger import logger

# class EvaluatorLLM:
#     """
#     Factory class for creating a native Ragas InstructorLLM instance using async clients.
#     """

#     def __init__(self) -> None:
#         logger.info(
#             "Initializing native Ragas evaluator model '%s'",
#             settings.evaluation_model,
#         )

#         # 1. Clean the model identifier string to match what your host target expects
#         # If the string contains 'openai/gpt-oss-120b', clean it or pass it directly 
#         # depending on your custom Groq cloud hosting deployment specifications.
#         target_model = settings.evaluation_model
        
#         # 2. Force Groq Cloud infrastructure execution via AsyncOpenAI client architecture
#         # Ragas factory uses AsyncOpenAI natively for generic custom API gateways
#         from openai import AsyncOpenAI
        
#         logger.info("Routing evaluation tokens through Groq Cloud API Gateway...")
#         native_client = AsyncOpenAI(
#             api_key=settings.groq_api_key,
#             base_url="https://api.groq.com/openai/v1"  # Force client to point to Groq instead of OpenAI
#         )

#         # 3. Build the Ragas framework Instructor model interface
#         self._llm = llm_factory(
#             model=target_model,
#             client=native_client
#         )

#         logger.info("Loading local HuggingFace Embedding: sentence-transformers/all-MiniLM-L6-v2...")

#         self._embeddings = HuggingFaceEmbeddings(
#             model="sentence-transformers/all-MiniLM-L6-v2"
#         )
        
#         logger.info("Native Ragas embedding engine initialized successfully.")

#     @property
#     def llm(self):
#         """
#         Return the native Ragas InstructorLLM engine.
#         """
#         return self._llm

#     @property
#     def embeddings(self):
#         """Return the native Ragas Embedding engine."""
#         return self._embeddings


from openai import AsyncOpenAI

from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.llms import llm_factory

from config.settings import settings
from utils.logger import logger


class EvaluatorLLM:
    """
    Creates the evaluator LLM and embedding model used by RAGAS.
    Routes requests through Groq's OpenAI-compatible endpoint.
    """

    def __init__(self) -> None:

        logger.info(
            "Initializing evaluator model: %s",
            settings.evaluation_model,
        )

        client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self._llm = llm_factory(
            model=settings.evaluation_model,
            client=client,
        )

        logger.info(
            "Loading evaluator embedding model..."
        )

        self._embeddings = RagasHuggingFaceEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info(
            "Evaluator initialized successfully."
        )

    @property
    def llm(self):
        return self._llm

    @property
    def embeddings(self):
        return self._embeddings
