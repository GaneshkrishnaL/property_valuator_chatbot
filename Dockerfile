# Dockerfile for Property Valuation RAG Chatbot
# Deploys to Google Cloud Run with full GCP integration:
#   - Document AI for PDF extraction (OCR, tables, forms)
#   - Vertex AI text-embedding-004 for semantic embeddings (768-dim)
#   - Gemini 2.0 Flash for LLM generation

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching — changes less often)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Data ingestion happens at RUNTIME (not build time) via startup.sh.
# Why? Because we need GCP credentials to call Vertex AI embeddings,
# and those credentials come from the Cloud Run service account —
# which is only available at runtime, not during Docker build.
#
# The startup.sh script:
#   1. Checks if ChromaDB exists
#   2. If not, runs python ingest.py (generates PDFs → extracts → embeds → stores)
#   3. Then starts Streamlit
#
# Cold start is ~30-60 seconds (one-time). Container stays warm after that.

EXPOSE 8080
CMD ["./startup.sh"]
