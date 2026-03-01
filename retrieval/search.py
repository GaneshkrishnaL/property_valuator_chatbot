"""
Hybrid search: metadata filtering + vector similarity + reranking.

ARCHITECTURAL NOTE — Why hybrid search?
Pure vector search can't handle structured queries like "3-bedroom homes
under $500K in Austin." The embedding might capture the semantic meaning
but won't reliably filter on exact numeric constraints. Our hybrid approach:

  1. METADATA FILTERING — Parse structured constraints from the query
     (bedrooms, price range, neighborhood) and apply them as exact filters
     in the vector store BEFORE running vector search. This dramatically
     reduces the search space and ensures results match hard constraints.

  2. VECTOR SIMILARITY — Cosine similarity on the filtered set captures
     semantic relevance that metadata alone can't express (e.g., "recently
     renovated" or "family-friendly neighborhood").

  3. RERANKING — After retrieval, we rerank by combining the vector
     similarity score with metadata match quality. A result that matches
     more metadata fields gets a boost, ensuring the most relevant
     results float to the top.

  4. COMPARISON SUPPORT — Queries like "Compare East Austin vs South Lamar"
     detect comparison intent, extract multiple neighborhoods, and use OR
     filters so results from both areas are returned.
"""

import logging
import re

from pydantic import BaseModel, Field

import config
from retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A single search result with score and provenance."""
    text: str
    score: float
    document_source: str = ""
    document_type: str = ""
    section_title: str = ""
    property_address: str = ""
    neighborhood: str = ""
    metadata: dict = Field(default_factory=dict)


class QueryFilters(BaseModel):
    """Structured filters extracted from a natural language query."""
    bedrooms: int | None = None
    bathrooms: float | None = None
    max_price: int | None = None
    min_price: int | None = None
    neighborhood: str | None = None
    neighborhoods: list[str] = Field(default_factory=list)
    is_comparison: bool = False
    document_type: str | None = None
    property_address: str | None = None


# Known neighborhoods in the dataset
_NEIGHBORHOODS = {
    "downtown austin": "Downtown Austin",
    "south congress": "South Congress",
    "east austin": "East Austin",
    "tarrytown": "Tarrytown",
    "south lamar": "South Lamar",
}

# Patterns that indicate a comparison query
_COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bbetween\b.*\band\b",
    r"\bdifference\s+between\b",
    r"\bhow\s+do(?:es)?\b.*\bcompare\b",
]


def _is_comparison_query(query_lower: str) -> bool:
    """Detect if the query is asking for a comparison between areas/properties."""
    return any(re.search(p, query_lower) for p in _COMPARISON_PATTERNS)


def _extract_neighborhoods(query_lower: str) -> list[str]:
    """Extract all matching neighborhoods from the query."""
    found = []
    for key, canonical in _NEIGHBORHOODS.items():
        if key in query_lower:
            found.append(canonical)
    return found


def extract_query_filters(query: str) -> QueryFilters:
    """
    Parse a natural language query to extract structured metadata filters.

    Handles both regular queries and comparison queries:
      "3-bedroom homes in Austin under $500K" → bedrooms=3, max_price=500000
      "Compare East Austin vs South Lamar"    → neighborhoods=["East Austin", "South Lamar"]
      "What's the value of 742 Evergreen Terrace?" → property_address match
    """
    filters = QueryFilters()
    query_lower = query.lower()

    # Bedrooms
    bed_match = re.search(r"(\d+)\s*[-\s]?bed(?:room)?s?", query_lower)
    if bed_match:
        filters.bedrooms = int(bed_match.group(1))

    # Bathrooms
    bath_match = re.search(r"(\d+\.?\d*)\s*[-\s]?bath(?:room)?s?", query_lower)
    if bath_match:
        filters.bathrooms = float(bath_match.group(1))

    # Price constraints
    under_match = re.search(r"(?:under|below|less\s+than|max|<)\s*\$?([\d,]+)\s*k?", query_lower)
    if under_match:
        price = int(under_match.group(1).replace(",", ""))
        filters.max_price = price * 1000 if price < 10000 else price

    over_match = re.search(r"(?:over|above|more\s+than|min|>)\s*\$?([\d,]+)\s*k?", query_lower)
    if over_match:
        price = int(over_match.group(1).replace(",", ""))
        filters.min_price = price * 1000 if price < 10000 else price

    # Neighborhood(s) — handle comparison vs single
    all_neighborhoods = _extract_neighborhoods(query_lower)

    if _is_comparison_query(query_lower) and len(all_neighborhoods) >= 2:
        filters.neighborhoods = all_neighborhoods
        filters.is_comparison = True
        logger.info(f"Comparison query detected: {all_neighborhoods}")
    elif all_neighborhoods:
        # Single neighborhood — use the first match
        filters.neighborhood = all_neighborhoods[0]

    # Document type hints
    if any(w in query_lower for w in ["market", "trend", "inventory", "average"]):
        filters.document_type = "MARKET_ANALYSIS"
    elif any(w in query_lower for w in ["comparable", "comp sales", "recent sales"]):
        filters.document_type = "COMPARABLE_SALES"
    elif any(w in query_lower for w in ["value", "appraisal", "assessed", "worth"]):
        filters.document_type = "APPRAISAL"
    elif any(w in query_lower for w in ["listing", "for sale", "mls"]):
        filters.document_type = "LISTING"

    # Specific property address
    addr_match = re.search(r"(\d+\s+[\w\s]+(?:street|st|avenue|ave|drive|dr|lane|ln|blvd|circle|terrace|way))", query_lower)
    if addr_match:
        filters.property_address = addr_match.group(1).strip()

    return filters


def build_where(filters: QueryFilters) -> dict | None:
    """
    Convert QueryFilters into a generic where clause.

    Works with both ChromaDB and Firestore backends. Supports:
      - Equality: {"field": value}
      - Range: {"field": {"$lte": val}} or {"$gte": val}
      - OR: {"field": {"$in": [val1, val2]}}
      - AND: {"$and": [condition1, condition2, ...]}
    """
    conditions: list[dict] = []

    if filters.bedrooms is not None:
        conditions.append({"bedrooms": filters.bedrooms})

    # Handle neighborhoods (multiple for comparisons, single for regular)
    if filters.neighborhoods:
        if len(filters.neighborhoods) == 1:
            conditions.append({"neighborhood": filters.neighborhoods[0]})
        else:
            conditions.append({"neighborhood": {"$in": filters.neighborhoods}})
    elif filters.neighborhood:
        conditions.append({"neighborhood": filters.neighborhood})

    if filters.document_type:
        conditions.append({"document_type": filters.document_type})
    if filters.max_price is not None:
        conditions.append({"price": {"$lte": filters.max_price}})
    if filters.min_price is not None:
        conditions.append({"price": {"$gte": filters.min_price}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def hybrid_search(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store,
    top_k: int = config.TOP_K_RESULTS,
) -> list[SearchResult]:
    """
    Execute hybrid search: metadata filter → vector search → rerank.

    Args:
        query: Natural language query from the user.
        embedding_model: Model to embed the query.
        vector_store: Vector store backend (ChromaDB or Firestore).
        top_k: Number of results to return.

    Returns:
        List of SearchResult objects sorted by relevance.
    """
    # Step 1: Extract structured filters from the query
    filters = extract_query_filters(query)
    logger.info(f"Extracted filters: {filters}")

    # Step 2: Build where clause
    where = build_where(filters)
    if where:
        logger.info(f"Applying metadata filter: {where}")

    # Step 3: Embed the query
    query_embedding = embedding_model.embed_query(query)

    # For comparison queries, request more results to ensure both neighborhoods
    # are represented in the final output
    multiplier = 3 if filters.is_comparison else 2

    # Step 4: Vector search with metadata filtering
    raw_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k * multiplier,
        where=where,
    )

    if not raw_results:
        # Fallback: try without filters in case filters were too restrictive
        logger.info("No results with filters, retrying without filters")
        raw_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * multiplier,
        )

    # Step 5: Rerank results
    reranked = _rerank(raw_results, filters, query)

    # Step 6: Filter by similarity threshold and take top-k
    results = []
    for r in reranked[:top_k]:
        if r["final_score"] >= config.SIMILARITY_THRESHOLD:
            meta = r.get("metadata", {})
            results.append(SearchResult(
                text=r["text"],
                score=r["final_score"],
                document_source=meta.get("document_source", ""),
                document_type=meta.get("document_type", ""),
                section_title=meta.get("section_title", ""),
                property_address=meta.get("property_address", ""),
                neighborhood=meta.get("neighborhood", ""),
                metadata=meta,
            ))

    logger.info(f"Returning {len(results)} results for: {query[:60]}...")
    return results


def _rerank(
    results: list[dict],
    filters: QueryFilters,
    query: str,
) -> list[dict]:
    """
    Rerank results by combining vector similarity with metadata match quality.

    For comparison queries, results from ANY of the target neighborhoods get
    a boost (not just one). This ensures both sides of a comparison are
    represented in the final output.
    """
    for r in results:
        base_score = r["score"]
        metadata_bonus = 0.0
        meta = r.get("metadata", {})

        # Bonus for matching specific address
        if filters.property_address:
            if filters.property_address.lower() in meta.get("property_address", "").lower():
                metadata_bonus += 0.15

        # Bonus for matching neighborhood(s)
        if filters.neighborhoods:
            # Comparison query: boost results from ANY target neighborhood
            doc_neighborhood = meta.get("neighborhood", "").lower()
            if any(n.lower() == doc_neighborhood for n in filters.neighborhoods):
                metadata_bonus += 0.08
        elif filters.neighborhood:
            if meta.get("neighborhood", "").lower() == filters.neighborhood.lower():
                metadata_bonus += 0.08

        # Bonus for matching document type
        if filters.document_type:
            if meta.get("document_type", "") == filters.document_type:
                metadata_bonus += 0.05

        # Bonus for matching bedroom count
        if filters.bedrooms is not None:
            if meta.get("bedrooms") == filters.bedrooms:
                metadata_bonus += 0.05

        # Compute final score (capped at 1.0)
        r["final_score"] = min(base_score + metadata_bonus, 1.0)

    # Sort by final score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
