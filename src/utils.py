"""
Utility functions for the RAG Document Q&A System.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


def generate_chunk_id(source: str, chunk_index: int) -> str:
    """
    Generate a deterministic unique ID for a document chunk.

    Args:
        source: Source document name.
        chunk_index: Index of the chunk within the document.

    Returns:
        A unique string ID.
    """
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_timestamp() -> str:
    """Get the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def setup_logging(level: int = logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Timer:
    """Simple context-manager timer for measuring operation durations."""

    def __init__(self, label: str = "Operation"):
        self.label = label
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.label} completed in {format_elapsed_time(self.elapsed)}")
