"""
Configuration module for RAG Document Q&A System.
Manages environment variables, model settings, and application constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploaded_documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
EDA_DIR = BASE_DIR / "eda"
VISUALIZATION_DIR = EDA_DIR / "visualizations"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Configuration ───────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else "none"))

# ─── Embedding Model Configuration ───────────────────────────
# Using a local HuggingFace model — no API key required
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── LLM Configuration ───────────────────────────────────────
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
LLM_MODEL_NAME = GEMINI_MODEL_NAME if LLM_PROVIDER == "gemini" else OPENAI_MODEL_NAME
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ─── Text Splitting Configuration ────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ─── Retrieval Configuration ─────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ─── ChromaDB Configuration ──────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_documents"

# ─── Supported File Types ────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# ─── Application Metadata ────────────────────────────────────
APP_TITLE = "RAG Document Q&A System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "A Retrieval-Augmented Generation system for querying "
    "enterprise documents using LangChain, ChromaDB, and OpenAI."
)
