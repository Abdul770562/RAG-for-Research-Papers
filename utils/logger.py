import logging
from pathlib import Path

from config.settings import LOG_DIR


LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "rag.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("RAG")