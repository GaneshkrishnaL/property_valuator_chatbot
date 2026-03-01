"""
Vector store abstraction with ChromaDB (local) and Firestore (GCP) backends.

Backend selection:
  - USE_FIRESTORE=true  → Firestore Vector Search (production on GCP)
  - USE_FIRESTORE=false → ChromaDB PersistentClient (local development)

Use get_vector_store() factory to get the right backend based on config.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

import config
from pipeline.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with rich metadata support."""

    def __init__(
        self,
        persist_dir: str | Path = config.CHROMA_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
    ):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at {self._persist_dir}")
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        logger.info(f"Collection '{collection_name}' has {self._collection.count()} documents")

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add document chunks with their embeddings and metadata to the store.

        ChromaDB stores embeddings, documents (text), and metadata together,
        enabling both vector similarity search AND metadata filtering.
        """
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata_dict for c in chunks]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = config.TOP_K_RESULTS,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """
        Search the vector store with optional metadata filtering.

        This is the core of hybrid search:
          1. 'where' filters on metadata BEFORE vector search
          2. Vector similarity ranks the filtered results
          3. Caller can then rerank the results

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            where: ChromaDB metadata filter (e.g., {"bedrooms": 3}).
            where_document: Full-text filter on document content.

        Returns:
            List of dicts with 'text', 'metadata', 'score', 'chunk_id'.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            # If filtered search fails (e.g., no matches), try without filters
            logger.warning(f"Filtered search failed ({e}), falling back to unfiltered")
            kwargs.pop("where", None)
            kwargs.pop("where_document", None)
            results = self._collection.query(**kwargs)

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns distances; convert to similarity score
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine distance → similarity

                formatted.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(similarity, 4),
                })

        return formatted

    def reset(self) -> None:
        """Delete the collection and recreate it (for re-ingestion)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset")


def get_vector_store():
    """
    Factory: returns the right vector store backend based on config.

    - USE_FIRESTORE=true  → FirestoreVectorStore (GCP production)
    - USE_FIRESTORE=false → VectorStore (ChromaDB local)
    """
    if config.USE_FIRESTORE:
        from retrieval.firestore_vector_store import FirestoreVectorStore
        return FirestoreVectorStore()
    return VectorStore()
