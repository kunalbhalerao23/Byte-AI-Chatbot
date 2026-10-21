"""Central configuration. All secrets/settings come from environment variables."""
import os

from dotenv import load_dotenv

# Reads a local .env file (git-ignored) into the process environment.
load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REQUEST_TIMEOUT_SECONDS = _float("REQUEST_TIMEOUT_SECONDS", 30.0)
MAX_RETRIES = _int("MAX_RETRIES", 2)

# Context window management
MAX_RECENT_MESSAGES = _int("MAX_RECENT_MESSAGES", 10)
SUMMARIZE_BATCH = _int("SUMMARIZE_BATCH", 4)

# Input limits
MAX_MESSAGE_CHARS = 4000
MAX_CONVERSATIONS = 1000  # in-memory store cap; oldest are evicted
