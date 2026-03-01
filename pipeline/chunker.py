"""
Semantic chunking with rich metadata for real estate documents.

ARCHITECTURAL NOTE — Why semantic chunking over fixed-size?
Fixed-size chunking (e.g., 500 chars) can split a property's details across
two chunks, leading to incomplete retrieval. For example, a property's address
might end up in chunk 1 while its price lands in chunk 2. Semantic chunking
ensures each property's data stays together as a meaningful unit.

Each chunk carries rich metadata that enables hybrid search:
  - document_type: filter by appraisal vs. listing vs. market analysis
  - property_address: filter by specific property
  - price_range: filter "homes under $500K"
  - neighborhood: filter by area
  - bedrooms/bathrooms: filter by features
"""

import hashlib
import logging
from typing import Any

from pydantic import BaseModel, Field

from pipeline.document_classifier import DocumentType

logger = logging.getLogger(__name__)

# Target chunk size in characters (roughly 500-800 tokens)
TARGET_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200


class DocumentChunk(BaseModel):
    """A single chunk of text with rich metadata for hybrid search."""
    chunk_id: str = Field(description="Unique ID for this chunk")
    text: str = Field(description="The actual text content for embedding")
    document_source: str = Field(description="Source PDF filename")
    document_type: str = Field(description="APPRAISAL, LISTING, MARKET_ANALYSIS, etc.")
    page_number: int = Field(default=1, description="Source page in the PDF")
    section_title: str = Field(default="", description="Section this chunk belongs to")

    # Structured metadata for filtering (populated from parser output)
    property_address: str = Field(default="")
    neighborhood: str = Field(default="")
    bedrooms: int | None = Field(default=None)
    bathrooms: float | None = Field(default=None)
    square_footage: int | None = Field(default=None)
    price: int | None = Field(default=None, description="Assessed, listed, or sale price")
    year_built: int | None = Field(default=None)

    @property
    def metadata_dict(self) -> dict[str, Any]:
        """Return metadata as a flat dict for ChromaDB storage."""
        meta = {
            "document_source": self.document_source,
            "document_type": self.document_type,
            "section_title": self.section_title,
            "property_address": self.property_address,
            "neighborhood": self.neighborhood,
        }
        if self.bedrooms is not None:
            meta["bedrooms"] = self.bedrooms
        if self.bathrooms is not None:
            meta["bathrooms"] = self.bathrooms
        if self.square_footage is not None:
            meta["square_footage"] = self.square_footage
        if self.price is not None:
            meta["price"] = self.price
        if self.year_built is not None:
            meta["year_built"] = self.year_built
        return meta


def chunk_document(
    filename: str,
    document_type: str,
    parsed_data: dict[str, Any],
) -> list[DocumentChunk]:
    """
    Create semantically meaningful chunks from a parsed document.

    Strategy:
      1. Each text_section from the parser becomes one or more chunks.
      2. Structured fields are attached as metadata to every chunk
         from that document (enabling metadata filtering).
      3. If a section is too long, we split on paragraph boundaries
         with overlapping context to preserve coherence.

    Args:
        filename: Source PDF filename.
        document_type: Classification result (APPRAISAL, LISTING, etc.).
        parsed_data: Output from the appropriate custom parser.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    structured = parsed_data.get("structured_fields", {})
    text_sections = parsed_data.get("text_sections", [])

    # Build shared metadata from structured fields
    shared_meta = _build_shared_metadata(structured, document_type)

    chunks: list[DocumentChunk] = []

    # If the parser found no text sections, create a single chunk
    if not text_sections:
        logger.warning(f"No text sections found for {filename}, creating minimal chunk")
        chunk = DocumentChunk(
            chunk_id=_make_chunk_id(filename, "full", 0),
            text=f"Document: {filename}. Type: {document_type}.",
            document_source=filename,
            document_type=document_type,
            section_title="Full Document",
            **shared_meta,
        )
        return [chunk]

    for section in text_sections:
        title = section.get("title", "")
        text = section.get("text", "").strip()

        if not text:
            continue

        # If text fits in one chunk, keep it together (semantic boundary)
        if len(text) <= TARGET_CHUNK_CHARS:
            chunk = DocumentChunk(
                chunk_id=_make_chunk_id(filename, title, 0),
                text=f"[{title}]\n{text}",
                document_source=filename,
                document_type=document_type,
                section_title=title,
                **shared_meta,
            )
            chunks.append(chunk)
        else:
            # Split on paragraph boundaries with overlap
            sub_chunks = _split_with_overlap(text, title)
            for i, sub_text in enumerate(sub_chunks):
                chunk = DocumentChunk(
                    chunk_id=_make_chunk_id(filename, title, i),
                    text=f"[{title}]\n{sub_text}",
                    document_source=filename,
                    document_type=document_type,
                    section_title=title,
                    **shared_meta,
                )
                chunks.append(chunk)

    # Add a structured-data-only chunk for property documents
    # This ensures pure metadata queries ("3 bed in Austin") can match
    if document_type in (DocumentType.APPRAISAL, DocumentType.LISTING):
        structured_text = _format_structured_fields(structured)
        if structured_text:
            chunk = DocumentChunk(
                chunk_id=_make_chunk_id(filename, "structured_summary", 0),
                text=structured_text,
                document_source=filename,
                document_type=document_type,
                section_title="Property Summary",
                **shared_meta,
            )
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {filename}")
    return chunks


def _build_shared_metadata(structured: dict, document_type: str) -> dict[str, Any]:
    """Extract metadata fields from structured parser output."""
    meta: dict[str, Any] = {}

    meta["property_address"] = structured.get("property_address", "")
    meta["neighborhood"] = structured.get("neighborhood", "")
    meta["bedrooms"] = structured.get("bedrooms")
    meta["bathrooms"] = structured.get("bathrooms")
    meta["square_footage"] = structured.get("square_footage")
    meta["year_built"] = structured.get("year_built")

    # Determine the most relevant price
    price = (
        structured.get("listing_price")
        or structured.get("assessed_value")
        or structured.get("estimated_value")
        or structured.get("avg_sale_price")
    )
    meta["price"] = price

    return meta


def _split_with_overlap(text: str, section_title: str) -> list[str]:
    """Split text on paragraph boundaries with overlapping context."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    sub_chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > TARGET_CHUNK_CHARS and current_chunk:
            sub_chunks.append("\n\n".join(current_chunk))

            # Keep last paragraph as overlap for context continuity
            overlap = current_chunk[-1] if current_chunk else ""
            current_chunk = [overlap] if overlap else []
            current_len = len(overlap)

        current_chunk.append(para)
        current_len += len(para)

    if current_chunk:
        sub_chunks.append("\n\n".join(current_chunk))

    return sub_chunks


def _format_structured_fields(structured: dict) -> str:
    """Create a text summary of structured fields for embedding."""
    parts = []
    field_labels = {
        "property_address": "Property Address",
        "listing_price": "Listing Price",
        "assessed_value": "Assessed Value",
        "estimated_value": "Estimated Market Value",
        "square_footage": "Square Footage",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "lot_size": "Lot Size",
        "year_built": "Year Built",
        "neighborhood": "Neighborhood",
        "days_on_market": "Days on Market",
    }

    for field_key, label in field_labels.items():
        value = structured.get(field_key)
        if value:
            if isinstance(value, int) and value > 10000:
                parts.append(f"{label}: ${value:,}")
            else:
                parts.append(f"{label}: {value}")

    return "Property Summary:\n" + "\n".join(parts) if parts else ""


def _make_chunk_id(filename: str, section: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{filename}::{section}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
