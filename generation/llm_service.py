import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from config.settings import settings
from utils.logger import logger


class LLMService:
    """
    Handles communication with the LLM.

    Responsibilities
    ----------------
    - Initialize the LLM
    - Send prompts
    - Receive responses
    - Log inference information

    This class does NOT:
    --------------------
    - Retrieve documents
    - Build prompts
    - Format citations
    """

    def __init__(self) -> None:

        self.llm = ChatGroq(
            model=settings.groq_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.groq_api_key,
        )

    def generate(
    self,
    messages: list[BaseMessage],
    ) -> str:

        logger.info(
            "Sending request to Groq model '%s'",
            settings.groq_model,
        )

        start_time = time.perf_counter()

        response = self.llm.invoke(messages)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "LLM response received in %.2f sec",
            elapsed,
        )

        if isinstance(response.content, str):
            return response.content
        raise ValueError(f"Expected a string response from LLM, got: {type(response.content)}")