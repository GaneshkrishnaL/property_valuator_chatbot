"""
Type-specific parsers for structured field extraction.

ARCHITECTURAL NOTE — Why custom parsers over generic RAG?
Standard RAG tools treat all documents as flat text. A property appraisal
has specific fields (assessed value, square footage, bedrooms) that users
will query by. Custom parsers extract these as structured metadata, enabling
hybrid search: "3-bedroom homes under $500K" can filter on metadata BEFORE
doing expensive vector similarity search.

Each parser outputs:
  - Structured fields → become metadata for filtering in the vector store
  - Raw text sections → become the content for embedding and retrieval
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from pipeline.pdf_extractor import ExtractedDocument

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Base class for document-type-specific parsers."""

    @abstractmethod
    def parse(self, doc: ExtractedDocument) -> dict[str, Any]:
        """
        Parse a document and return structured fields + raw text sections.

        Returns:
            Dict with 'structured_fields' (metadata) and 'text_sections' (for embedding).
        """
        ...


class AppraisalParser(BaseParser):
    """Extract structured fields from property appraisal reports."""

    def parse(self, doc: ExtractedDocument) -> dict[str, Any]:
        text = doc.full_text
        structured = {
            "document_type": "APPRAISAL",
            "property_address": self._extract_address(text),
            "square_footage": self._extract_number(text, r"(\d{1,2},?\d{3})\s*(?:sq(?:uare)?\s*f(?:ee)?t|sq\s*ft)"),
            "bedrooms": self._extract_int(text, r"(\d+)\s*bedroom"),
            "bathrooms": self._extract_float(text, r"(\d+\.?\d*)\s*bathroom"),
            "lot_size": self._extract_pattern(text, r"([\d.]+\s*acres?)"),
            "year_built": self._extract_int(text, r"built\s+in\s+(\d{4})"),
            "assessed_value": self._extract_currency(text, r"assessed\s+value\s+(?:is\s+)?\$?([\d,]+)"),
            "estimated_value": self._extract_currency(text, r"estimated\s+market\s+value.*?\$?([\d,]+)"),
            "neighborhood": self._extract_neighborhood(text),
            "appraiser_name": self._extract_pattern(text, r"Appraiser:\s*([^|,\n]+)"),
        }

        # Extract comparable sales from tables
        comps = self._extract_comp_table(doc)
        structured["comparable_sales"] = comps

        text_sections = self._split_sections(text)

        logger.info(f"AppraisalParser extracted {sum(1 for v in structured.values() if v)} fields from {doc.filename}")
        return {"structured_fields": structured, "text_sections": text_sections}

    def _extract_comp_table(self, doc: ExtractedDocument) -> list[dict]:
        """Extract comparable sales from tables in the document."""
        comps = []
        for table in doc.all_tables:
            if not table:
                continue
            headers = [h.lower().strip() for h in table[0]]
            # Look for tables with price and address columns
            if any("price" in h or "sale" in h for h in headers) and any("address" in h for h in headers):
                for row in table[1:]:
                    if len(row) >= 3:
                        comp = {}
                        for i, header in enumerate(headers):
                            if i < len(row):
                                if "address" in header:
                                    comp["address"] = row[i]
                                elif "price" in header and "sq" not in header:
                                    comp["sale_price"] = row[i]
                                elif "date" in header:
                                    comp["sale_date"] = row[i]
                                elif "sq" in header and "price" not in header:
                                    comp["sqft"] = row[i]
                                elif "$/sq" in header or "per" in header:
                                    comp["price_per_sqft"] = row[i]
                        if comp:
                            comps.append(comp)
        return comps

    def _split_sections(self, text: str) -> list[dict[str, str]]:
        """Split document into meaningful sections for chunking."""
        sections = []
        current_title = "Introduction"
        current_text: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            # Detect section headers (ALL CAPS or bold-style)
            if stripped and (stripped.isupper() or stripped.endswith("Information") or
                          stripped.endswith("Assessment") or stripped.endswith("Analysis") or
                          stripped.endswith("Conclusion")):
                if current_text:
                    sections.append({"title": current_title, "text": "\n".join(current_text)})
                current_title = stripped
                current_text = []
            else:
                current_text.append(line)

        if current_text:
            sections.append({"title": current_title, "text": "\n".join(current_text)})

        return sections

    # ── helper methods ──

    def _extract_address(self, text: str) -> str:
        match = re.search(r"(?:located\s+at|address)\s*:?\s*([^.|\n]{10,60}(?:TX|Texas)\s*\d{5})", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_number(self, text: str, pattern: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
        return None

    def _extract_int(self, text: str, pattern: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_float(self, text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _extract_currency(self, text: str, pattern: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
        return None

    def _extract_pattern(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_neighborhood(self, text: str) -> str:
        match = re.search(r"(?:neighborhood|area)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
        if not match:
            # Try extracting from known neighborhoods
            neighborhoods = ["Downtown Austin", "South Congress", "East Austin", "Tarrytown", "South Lamar"]
            for n in neighborhoods:
                if n.lower() in text.lower():
                    return n
        return match.group(1).strip() if match else ""


class ListingParser(BaseParser):
    """Extract structured fields from MLS-style listing sheets."""

    def parse(self, doc: ExtractedDocument) -> dict[str, Any]:
        text = doc.full_text

        structured = {
            "document_type": "LISTING",
            "property_address": self._extract_listing_address(text),
            "listing_price": self._extract_currency(text, r"(?:list\s*price|price)\s*:?\s*\$?([\d,]+)"),
            "mls_number": self._extract_pattern(text, r"MLS\s*#?\s*(\S+)"),
            "bedrooms": self._extract_int(text, r"Bedrooms?\s*(\d+)"),
            "bathrooms": self._extract_int(text, r"Bathrooms?\s*(\d+)"),
            "square_footage": self._extract_number(text, r"Square\s*Feet?\s*([\d,]+)"),
            "lot_size": self._extract_pattern(text, r"Lot\s*Size\s*([^\n]+)"),
            "year_built": self._extract_int(text, r"Year\s*Built\s*(\d{4})"),
            "days_on_market": self._extract_int(text, r"Days\s+on\s+Market\s*:?\s*(\d+)"),
            "listing_date": self._extract_pattern(text, r"Listing\s+Date\s*:?\s*([^\n]+)"),
            "agent_info": self._extract_pattern(text, r"Listed\s+by\s*:?\s*([^\n]+)"),
            "neighborhood": self._extract_neighborhood(text),
            "property_features": self._extract_features(text),
        }

        text_sections = [
            {"title": "Listing Details", "text": text[:500]},
            {"title": "Agent Remarks", "text": self._extract_remarks(text)},
        ]

        logger.info(f"ListingParser extracted {sum(1 for v in structured.values() if v)} fields from {doc.filename}")
        return {"structured_fields": structured, "text_sections": text_sections}

    def _extract_listing_address(self, text: str) -> str:
        # Listings often have address on its own line after "Property Address"
        match = re.search(r"Property\s+Address\s*\n\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: look for Texas address pattern
        match = re.search(r"(\d+[^,\n]+,\s*Austin,?\s*TX\s*\d{5})", text)
        return match.group(1).strip() if match else ""

    def _extract_remarks(self, text: str) -> str:
        match = re.search(r"(?:Agent\s+Remarks|Description)\s*\n([\s\S]*?)(?=\n[A-Z][a-z]+\s+(?:Date|by)|$)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_features(self, text: str) -> list[str]:
        features = []
        patterns = [r"Garage\s+(.+?)(?:\n|$)", r"Heating\s+(.+?)(?:\n|$)", r"Cooling\s+(.+?)(?:\n|$)",
                     r"Flooring\s+(.+?)(?:\n|$)", r"Style\s+(.+?)(?:\n|$)"]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                features.append(match.group(1).strip())
        return features

    # Reuse helpers
    def _extract_currency(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1).replace(",", "")) if match else None

    def _extract_pattern(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_int(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_number(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1).replace(",", "")) if match else None

    def _extract_neighborhood(self, text):
        neighborhoods = ["Downtown Austin", "South Congress", "East Austin", "Tarrytown", "South Lamar"]
        for n in neighborhoods:
            if n.lower() in text.lower():
                return n
        return ""


class MarketAnalysisParser(BaseParser):
    """Extract structured fields from market analysis reports."""

    def parse(self, doc: ExtractedDocument) -> dict[str, Any]:
        text = doc.full_text

        # Market analysis has per-neighborhood data
        neighborhoods = self._extract_neighborhood_stats(doc)

        structured = {
            "document_type": "MARKET_ANALYSIS",
            "report_period": self._extract_pattern(text, r"Report\s+Period\s*:?\s*([^\n|]+)"),
            "neighborhoods": neighborhoods,
        }

        text_sections = self._split_by_neighborhood(text)
        if not text_sections:
            text_sections = [{"title": "Market Analysis", "text": text}]

        logger.info(f"MarketAnalysisParser extracted data for {len(neighborhoods)} neighborhoods from {doc.filename}")
        return {"structured_fields": structured, "text_sections": text_sections}

    def _extract_neighborhood_stats(self, doc: ExtractedDocument) -> list[dict]:
        """Extract per-neighborhood statistics from tables."""
        stats = []
        for table in doc.all_tables:
            if not table:
                continue
            headers = [h.lower().strip() for h in table[0]]
            if any("neighborhood" in h for h in headers):
                for row in table[1:]:
                    if len(row) >= 4:
                        entry = {}
                        for i, header in enumerate(headers):
                            if i < len(row):
                                if "neighborhood" in header:
                                    entry["neighborhood"] = row[i]
                                elif "avg" in header and "price" in header:
                                    entry["avg_price"] = row[i]
                                elif "median" in header:
                                    entry["median_price"] = row[i]
                                elif "dom" in header or "days" in header:
                                    entry["avg_days_on_market"] = row[i]
                                elif "inventory" in header:
                                    entry["inventory_count"] = row[i]
                                elif "yoy" in header or "change" in header:
                                    entry["price_trend"] = row[i]
                        if entry:
                            stats.append(entry)
        return stats

    def _split_by_neighborhood(self, text: str) -> list[dict[str, str]]:
        """Split the market analysis by neighborhood sections."""
        sections = []
        # Match "Neighborhood Name — Detailed Analysis" pattern
        parts = re.split(r"((?:Downtown Austin|South Congress|East Austin|Tarrytown|South Lamar)\s*(?:[-—]+|:)\s*Detailed\s+Analysis)", text, flags=re.IGNORECASE)

        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append({"title": title, "text": content})

        # Add executive summary if present
        exec_match = re.search(r"Executive\s+Summary\s*\n([\s\S]*?)(?=Neighborhood\s+Performance|$)", text, re.IGNORECASE)
        if exec_match:
            sections.insert(0, {"title": "Executive Summary", "text": exec_match.group(1).strip()})

        return sections

    def _extract_pattern(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""


class ComparableSalesParser(BaseParser):
    """Extract structured fields from comparable sales reports."""

    def parse(self, doc: ExtractedDocument) -> dict[str, Any]:
        text = doc.full_text
        sales = self._extract_sales_table(doc)

        structured = {
            "document_type": "COMPARABLE_SALES",
            "sales_period": self._extract_pattern(text, r"Sales?\s+Period\s*:?\s*([^\n|]+)"),
            "sales": sales,
            "total_sales_count": len(sales),
        }

        # Compute aggregate stats if we have data
        if sales:
            prices = [s.get("sale_price_numeric", 0) for s in sales if s.get("sale_price_numeric")]
            if prices:
                structured["avg_sale_price"] = int(sum(prices) / len(prices))
                structured["min_sale_price"] = min(prices)
                structured["max_sale_price"] = max(prices)

        text_sections = [
            {"title": "Comparable Sales Data", "text": text[:1000]},
        ]
        # Add statistical summary if present
        stats_match = re.search(r"Statistical\s+Summary\s*\n([\s\S]+)", text, re.IGNORECASE)
        if stats_match:
            text_sections.append({"title": "Statistical Summary", "text": stats_match.group(1).strip()})

        logger.info(f"ComparableSalesParser extracted {len(sales)} sales from {doc.filename}")
        return {"structured_fields": structured, "text_sections": text_sections}

    def _extract_sales_table(self, doc: ExtractedDocument) -> list[dict]:
        """Extract individual sales from tables."""
        sales = []
        for table in doc.all_tables:
            if not table:
                continue
            headers = [h.lower().strip() for h in table[0]]
            if any("address" in h for h in headers) and any("price" in h for h in headers):
                for row in table[1:]:
                    if len(row) >= 3:
                        sale: dict[str, Any] = {}
                        for i, header in enumerate(headers):
                            if i < len(row):
                                val = row[i]
                                if "address" in header:
                                    sale["address"] = val
                                elif "price" in header and "sq" not in header and "per" not in header:
                                    sale["sale_price"] = val
                                    # Parse numeric value for metadata filtering
                                    numeric = re.sub(r"[^\d]", "", val)
                                    if numeric:
                                        sale["sale_price_numeric"] = int(numeric)
                                elif "date" in header:
                                    sale["sale_date"] = val
                                elif "sq" in header and "price" not in header:
                                    sale["sqft"] = val
                                elif "$/sq" in header or "per" in header:
                                    sale["price_per_sqft"] = val
                        if sale:
                            sales.append(sale)
        return sales

    def _extract_pattern(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""


# ── Parser registry ──

PARSERS: dict[str, BaseParser] = {
    "APPRAISAL": AppraisalParser(),
    "LISTING": ListingParser(),
    "MARKET_ANALYSIS": MarketAnalysisParser(),
    "COMPARABLE_SALES": ComparableSalesParser(),
}


def get_parser(document_type: str) -> BaseParser | None:
    """Get the appropriate parser for a document type."""
    return PARSERS.get(document_type)
