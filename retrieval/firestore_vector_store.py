"""
Firestore Vector Search backend for the property valuation chatbot.

Replaces ChromaDB with Google Cloud Firestore's native vector search,
adding another GCP service to the stack and enabling serverless,
managed vector storage with built-in metadata filtering.

Key Firestore Vector Search features:
  - Native vector field type with cosine/euclidean/dot-product distance
  - Pre-filtering: chain .where() before .find_nearest() for metadata filters
  - Supports 'in' operator for OR-style filters (e.g., multiple neighborhoods)
  - Max 2048 embedding dimensions (we use 768 from text-embedding-004)
  - Serverless — no index endpoints to provision or manage
  - Auto-scales, zero config, integrates with Cloud Run service account
"""

import logging
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import config
from pipeline.chunker import DocumentChunk

logger = logging.getLogger(__name__)

# Firestore batch write limit
_BATCH_LIMIT = 500


class FirestoreVectorStore:
    """Firestore-backed vector store with native vector search and metadata filtering."""

    def __init__(
        self,
        project_id: str = config.GCP_PROJECT_ID,
        collection_name: str | None = None,
    ):
        self._project_id = project_id
        self._collection_name = collection_name or config.FIRESTORE_COLLECTION_NAME

        logger.info(f"Initializing Firestore vector store (project={project_id}, "
                     f"collection={self._collection_name})")
        self._client = firestore.Client(project=project_id)
        self._collection = self._client.collection(self._collection_name)

    @property
    def count(self) -> int:
        """Count documents in the collection (uses aggregation query)."""
        from google.cloud.firestore_v1 import aggregation
        agg_query = aggregation.AggregationQuery(self._collection)
        agg_query.count(alias="total")
        results = agg_query.get()
        for result in results:
            return result[0].value
        return 0

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks with embeddings and metadata in Firestore.

        Each chunk becomes a Firestore document with:
          - 'embedding': Vector field for similarity search
          - 'text': The chunk text content
          - Flat metadata fields (neighborhood, bedrooms, price, etc.)
            stored at top level for efficient filtering
        """
        if not chunks:
            return

        # Batch writes for efficiency (Firestore limit: 500 per batch)
        batch = self._client.batch()
        batch_count = 0

        for chunk, embedding in zip(chunks, embeddings):
            doc_ref = self._collection.document(chunk.chunk_id)

            # Build document data with flat metadata fields for filtering
            doc_data = {
                "text": chunk.text,
                "embedding": Vector(embedding),
                # Flat metadata fields — stored at top level so Firestore
                # where() filters can reference them directly
                "document_source": chunk.document_source,
                "document_type": chunk.document_type,
                "section_title": chunk.section_title,
                "page_number": chunk.page_number,
                "property_address": chunk.property_address,
                "neighborhood": chunk.neighborhood,
                "bedrooms": chunk.bedrooms if chunk.bedrooms is not None else 0,
                "bathrooms": chunk.bathrooms if chunk.bathrooms is not None else 0.0,
                "square_footage": chunk.square_footage if chunk.square_footage is not None else 0,
                "price": chunk.price if chunk.price is not None else 0,
                "year_built": chunk.year_built if chunk.year_built is not None else 0,
            }

            batch.set(doc_ref, doc_data)
            batch_count += 1

            # Commit when batch is full
            if batch_count >= _BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                batch_count = 0

        # Commit remaining
        if batch_count > 0:
            batch.commit()

        logger.info(f"Added {len(chunks)} chunks to Firestore collection "
                     f"'{self._collection_name}'")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = config.TOP_K_RESULTS,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """
        Search Firestore with vector similarity + optional metadata filtering.

        Uses Firestore's find_nearest() for KNN vector search with optional
        pre-filtering via where() clauses. The 'where' parameter uses the same
        generic format as ChromaDB for compatibility:
          - {"field": value}           → equality filter
          - {"field": {"$in": [...]}}  → OR filter (Firestore 'in' operator)
          - {"field": {"$lte": val}}   → less-than-or-equal
          - {"field": {"$gte": val}}   → greater-than-or-equal
          - {"$and": [...conditions]}  → AND of multiple conditions

        Args:
            query_embedding: The query vector (768-dim).
            top_k: Number of results to return.
            where: Metadata filter dict (generic format, translated to Firestore).
            where_document: Not used (ChromaDB compat placeholder).

        Returns:
            List of dicts with 'text', 'metadata', 'score', 'chunk_id'.
        """
        try:
            # Build query with optional metadata filters
            query = self._collection
            if where:
                query = self._apply_where(query, where)

            # Execute vector search
            vector_query = query.find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_embedding),
                distance_measure=DistanceMeasure.COSINE,
                limit=top_k,
                distance_result_field="vector_distance",
            )

            results = vector_query.get()
            return self._format_results(results)

        except Exception as e:
            logger.warning(f"Filtered search failed ({e}), retrying without filters")
            try:
                vector_query = self._collection.find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_embedding),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=top_k,
                    distance_result_field="vector_distance",
                )
                results = vector_query.get()
                return self._format_results(results)
            except Exception as e2:
                logger.error(f"Unfiltered search also failed: {e2}")
                return []

    def reset(self) -> None:
        """Delete all documents in the collection (for re-ingestion)."""
        docs = self._collection.list_documents()
        batch = self._client.batch()
        count = 0
        for doc_ref in docs:
            batch.delete(doc_ref)
            count += 1
            if count >= _BATCH_LIMIT:
                batch.commit()
                batch = self._client.batch()
                count = 0
        if count > 0:
            batch.commit()
        logger.info(f"Firestore collection '{self._collection_name}' reset")

    def _apply_where(self, query, where: dict):
        """
        Translate generic where dict to Firestore query filters.

        Handles: equality, $in, $lte, $gte, $and compound conditions.
        """
        if "$and" in where:
            # Compound filter: chain all conditions
            for condition in where["$and"]:
                query = self._apply_where(query, condition)
            return query

        for field, value in where.items():
            if field.startswith("$"):
                continue  # Skip operators at top level

            if isinstance(value, dict):
                # Operator-based filter
                if "$in" in value:
                    query = query.where(field, "in", value["$in"])
                elif "$lte" in value:
                    query = query.where(field, "<=", value["$lte"])
                elif "$gte" in value:
                    query = query.where(field, ">=", value["$gte"])
            else:
                # Simple equality
                query = query.where(field, "==", value)

        return query

    def _format_results(self, results) -> list[dict]:
        """Convert Firestore query results to the standard format."""
        formatted = []
        for doc_snapshot in results:
            data = doc_snapshot.to_dict()

            # Firestore COSINE distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1.0 = identical, 0.0 = orthogonal
            distance = data.get("vector_distance", 1.0)
            similarity = 1.0 - (distance / 2.0)

            # Build metadata dict (same structure as ChromaDB version)
            metadata = {
                "document_source": data.get("document_source", ""),
                "document_type": data.get("document_type", ""),
                "section_title": data.get("section_title", ""),
                "page_number": data.get("page_number", 1),
                "property_address": data.get("property_address", ""),
                "neighborhood": data.get("neighborhood", ""),
                "bedrooms": data.get("bedrooms", 0),
                "bathrooms": data.get("bathrooms", 0.0),
                "square_footage": data.get("square_footage", 0),
                "price": data.get("price", 0),
                "year_built": data.get("year_built", 0),
            }

            formatted.append({
                "chunk_id": doc_snapshot.id,
                "text": data.get("text", ""),
                "metadata": metadata,
                "score": round(similarity, 4),
            })

        return formatted
