#!/bin/bash
# Cloud Run startup script
# Runs data ingestion on first boot, then starts Streamlit.
#
# Why at startup and not at build time?
# Cloud Run containers are stateless — filesystem resets between deployments.
# We need Vertex AI embeddings (768-dim) for both ingestion AND query time.
# GCP credentials aren't available during Docker build, but they ARE available
# at runtime via the Cloud Run service account. So we ingest here.
#
# Cold start takes ~30-60 seconds (one-time), then the container stays warm.

set -e

echo "=== Property Valuation RAG Chatbot - Starting ==="

# Determine which vector store backend to use
if [ "${USE_FIRESTORE:-true}" = "true" ]; then
    MARKER_FILE="/app/.ingestion_complete"
    echo ">>> Using Firestore Vector Search backend"
else
    MARKER_FILE="/app/chroma_db/chroma.sqlite3"
    echo ">>> Using ChromaDB (local) backend"
fi

# Run ingestion if not already done in this container instance
if [ ! -f "$MARKER_FILE" ]; then
    echo ">>> Running data ingestion pipeline..."
    python ingest.py
    # Create marker file so we don't re-ingest on warm restarts
    touch /app/.ingestion_complete
    echo ">>> Ingestion complete!"
else
    echo ">>> Data already ingested, skipping."
fi

echo ">>> Starting Streamlit on port 8080..."
exec streamlit run demo/streamlit_app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
