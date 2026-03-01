"""
Document classifier for real estate PDFs.

Classifies each document into a type so the correct specialized parser
can handle extraction. Uses keyword matching and structural analysis
rather than ML — keeps the prototype simple and explainable.

In production, you could fine-tune a Gemini classifier or use Document AI's
custom classification feature for higher accuracy on edge cases.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from pipeline.pdf_extractor import ExtractedDocument

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    APPRAISAL = "APPRAISAL"
    LISTING = "LISTING"
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    COMPARABLE_SALES = "COMPARABLE_SALES"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassificationResult:
    """Result of document classification with confidence score."""
    document_type: DocumentType
    confidence: float   # 0.0 to 1.0
    reasoning: str      # human-readable explanation of the classification


# Keyword signals for each document type, weighted by specificity
_CLASSIFICATION_SIGNALS: dict[DocumentType, list[tuple[str, float]]] = {
    DocumentType.APPRAISAL: [
        (r"appraisal\s+report", 0.35),
        (r"appraiser", 0.20),
        (r"assessed\s+value", 0.15),
        (r"valuation\s+conclusion", 0.20),
        (r"subject\s+property", 0.10),
        (r"license\s*#", 0.10),
        (r"comparable\s+sales\s+analysis", 0.10),
    ],
    DocumentType.LISTING: [
        (r"property\s+listing", 0.30),
        (r"mls\s*#", 0.25),
        (r"list\s+price", 0.20),
        (r"agent\s+remarks", 0.15),
        (r"days\s+on\s+market", 0.10),
        (r"listing\s+date", 0.10),
    ],
    DocumentType.MARKET_ANALYSIS: [
        (r"market\s+analysis", 0.35),
        (r"market\s+trends?", 0.15),
        (r"year.over.year", 0.15),
        (r"inventory\s+level", 0.10),
        (r"executive\s+summary", 0.10),
        (r"neighborhood\s+performance", 0.15),
        (r"avg\s+dom|average\s+days\s+on\s+market", 0.10),
    ],
    DocumentType.COMPARABLE_SALES: [
        (r"comparable\s+sales\s+report", 0.35),
        (r"recent\s+comparable\s+sales", 0.20),
        (r"price\s+per\s+square\s+foot", 0.15),
        (r"statistical\s+summary", 0.10),
        (r"sales\s+period", 0.10),
        (r"\$/sq\s*ft", 0.15),
    ],
}


def classify_document(doc: ExtractedDocument) -> ClassificationResult:
    """
    Classify a document based on keyword signals and structural features.

    The classifier scores each document type by matching weighted regex
    patterns against the full text. Structural features (presence of tables,
    page count) provide secondary signals.

    Args:
        doc: Extracted document content.

    Returns:
        ClassificationResult with type, confidence, and reasoning.
    """
    text_lower = doc.full_text.lower()
    scores: dict[DocumentType, float] = {}
    matched_signals: dict[DocumentType, list[str]] = {}

    for doc_type, signals in _CLASSIFICATION_SIGNALS.items():
        total_score = 0.0
        matches = []
        for pattern, weight in signals:
            if re.search(pattern, text_lower):
                total_score += weight
                matches.append(pattern)
        scores[doc_type] = total_score
        matched_signals[doc_type] = matches

    # Structural bonus: appraisals typically have 2+ pages, listings are single page
    if doc.page_count >= 2:
        scores[DocumentType.APPRAISAL] += 0.05
        scores[DocumentType.MARKET_ANALYSIS] += 0.05

    # Structural bonus: lots of tables suggest comp sales or market analysis
    table_count = sum(1 for p in doc.pages if p.has_tables)
    if table_count >= 2:
        scores[DocumentType.COMPARABLE_SALES] += 0.05
        scores[DocumentType.MARKET_ANALYSIS] += 0.05

    # Find the winner
    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_type]

    # Normalize confidence to 0-1 range (max theoretical ~1.1)
    confidence = min(best_score, 1.0)

    # Fall back to UNKNOWN if confidence is too low
    if confidence < 0.2:
        best_type = DocumentType.UNKNOWN
        reasoning = "No strong signals matched any document type."
    else:
        matched = matched_signals[best_type]
        reasoning = (
            f"Classified as {best_type.value} based on {len(matched)} keyword signals: "
            f"{', '.join(matched[:3])}. Score: {best_score:.2f}"
        )

    result = ClassificationResult(
        document_type=best_type,
        confidence=round(confidence, 3),
        reasoning=reasoning,
    )

    logger.info(f"{doc.filename} → {result.document_type.value} (confidence: {result.confidence})")
    return result
