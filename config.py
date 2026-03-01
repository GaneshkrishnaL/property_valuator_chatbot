"""
Configuration for the Real Estate RAG Prototype.

Switch between Google Cloud and local backends using the flags below.
For the live demo, set USE_VERTEX_AI=False and USE_GEMINI=False to run
entirely with local/OpenAI fallbacks — no cloud dependency during the interview.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db")))

# ──────────────────────────────────────────────
# Backend selection flags
# Read from environment variables so Cloud Run can override them.
# Cloud Run deploy command sets these as env vars, e.g.:
#   --set-env-vars USE_VERTEX_AI=true,USE_GEMINI=true
# Locally, change these defaults or export the env vars.
# ──────────────────────────────────────────────
USE_VERTEX_AI = os.getenv("USE_VERTEX_AI", "true").lower() == "true"
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "true").lower() == "true"

# ──────────────────────────────────────────────
# Google Cloud settings (primary — production path)
# ──────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "property-valuation-chatbot")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_EMBEDDING_MODEL = "text-embedding-004"
GEMINI_MODEL = "gemini-2.0-flash"   # faster + cheaper than 1.5-pro, great for RAG

# Document AI processor (created in GCP console)
# NOTE: Document AI uses multi-region locations ("us", "eu") — NOT specific
# regions like "us-central1". This is different from Vertex AI / Cloud Run.
DOCUMENT_AI_LOCATION = os.getenv("DOCUMENT_AI_LOCATION", "us")
DOCUMENT_AI_PROCESSOR_ID = os.getenv("DOCUMENT_AI_PROCESSOR_ID", "e1e14121d4d81e8b")

# ──────────────────────────────────────────────
# Local / fallback settings
# ──────────────────────────────────────────────
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# ──────────────────────────────────────────────
# Vector store settings
# ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "real_estate_docs"
FIRESTORE_COLLECTION_NAME = os.getenv("FIRESTORE_COLLECTION", "real_estate_documents")
TOP_K_RESULTS = 5
CHUNK_SIZE_TOKENS = 600          # target tokens per chunk
CHUNK_OVERLAP_TOKENS = 80        # overlap between consecutive chunks
SIMILARITY_THRESHOLD = 0.3       # minimum cosine similarity to include

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
