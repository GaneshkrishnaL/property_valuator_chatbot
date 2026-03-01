"""
PDF text and table extraction using PyMuPDF (local) or Google Document AI (cloud).

ARCHITECTURAL NOTE — Why not LangChain's PyPDFLoader?
Standard RAG tools like LangChain's PyPDFLoader dump raw text without preserving
table structure or extracting structured fields. For real estate documents, losing
table structure means losing comparable sales data, which is critical for valuations.
Our custom extractor preserves table layouts and extracts metadata that becomes
searchable attributes in the vector store.

DUAL BACKEND:
  - LOCAL (USE_VERTEX_AI=False): PyMuPDF (fitz) — fast, no cloud dependency
  - CLOUD (USE_VERTEX_AI=True):  Google Document AI — OCR, table detection,
    form field extraction, entity detection, multilingual support
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import config

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Extracted content from a single PDF page."""
    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    has_tables: bool = False


@dataclass
class ExtractedDocument:
    """Full extraction result for a PDF document."""
    source_path: str
    filename: str
    page_count: int
    pages: list[ExtractedPage]
    full_text: str
    metadata: dict

    @property
    def all_tables(self) -> list[list[list[str]]]:
        tables = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables


def extract_pdf(pdf_path: Path) -> ExtractedDocument:
    """
    Extract text, tables, and metadata from a PDF file.

    Routes to the appropriate backend:
      - USE_VERTEX_AI=True  → Google Document AI (OCR, tables, entities)
      - USE_VERTEX_AI=False → PyMuPDF local extraction

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ExtractedDocument with page-level content and metadata.
    """
    if config.USE_VERTEX_AI:
        logger.info(f"Extracting with Document AI: {pdf_path.name}")
        return _extract_with_document_ai(pdf_path)
    else:
        logger.info(f"Extracting with PyMuPDF: {pdf_path.name}")
        return _extract_with_pymupdf(pdf_path)


def _extract_with_pymupdf(pdf_path: Path) -> ExtractedDocument:
    """Local extraction using PyMuPDF (fitz)."""
    import fitz

    doc = fitz.open(str(pdf_path))

    pages: list[ExtractedPage] = []
    all_text_parts: list[str] = []

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text")
            text = _filter_headers_footers(text)
            tables = _extract_tables_pymupdf(page)

            extracted_page = ExtractedPage(
                page_number=page_num + 1,
                text=text.strip(),
                tables=tables,
                has_tables=len(tables) > 0,
            )
            pages.append(extracted_page)
            all_text_parts.append(text.strip())

        except Exception as e:
            logger.warning(f"Failed to extract page {page_num + 1} from {pdf_path.name}: {e}")
            pages.append(ExtractedPage(page_number=page_num + 1, text="", tables=[]))

    meta = doc.metadata or {}
    metadata = {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "creation_date": meta.get("creationDate", ""),
        "page_count": len(doc),
        "filename": pdf_path.name,
    }
    doc.close()

    result = ExtractedDocument(
        source_path=str(pdf_path),
        filename=pdf_path.name,
        page_count=len(pages),
        pages=pages,
        full_text="\n\n".join(all_text_parts),
        metadata=metadata,
    )
    logger.info(
        f"Extracted {pdf_path.name}: {result.page_count} pages, "
        f"{sum(1 for p in pages if p.has_tables)} pages with tables"
    )
    return result


def _extract_with_document_ai(pdf_path: Path) -> ExtractedDocument:
    """
    Production extraction using Google Document AI.

    Provides OCR for scanned docs, ML-based table extraction,
    form field detection, and entity extraction (addresses, currencies).
    """
    from google.cloud import documentai_v1 as documentai

    # Document AI uses multi-region endpoints ("us", "eu"), not specific regions.
    # We must set the API endpoint to match the processor's location.
    api_endpoint = f"{config.DOCUMENT_AI_LOCATION}-documentai.googleapis.com"
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )

    # Read the PDF file
    with open(pdf_path, "rb") as f:
        content = f.read()

    # Build the processor resource path
    resource_name = client.processor_path(
        config.GCP_PROJECT_ID,
        config.DOCUMENT_AI_LOCATION,
        config.DOCUMENT_AI_PROCESSOR_ID,
    )

    raw_document = documentai.RawDocument(
        content=content, mime_type="application/pdf"
    )
    request = documentai.ProcessRequest(
        name=resource_name, raw_document=raw_document
    )

    result = client.process_document(request=request)
    document = result.document

    # Convert Document AI output to our ExtractedDocument format
    pages: list[ExtractedPage] = []
    all_text_parts: list[str] = []

    for page_idx, page in enumerate(document.pages):
        # Get text for this page using the text anchors
        page_text = _get_page_text(page, document.text)

        # Extract tables from this page
        tables = []
        for table in page.tables:
            extracted_table = _extract_table_from_docai(table, document.text)
            if extracted_table:
                tables.append(extracted_table)

        extracted_page = ExtractedPage(
            page_number=page_idx + 1,
            text=page_text.strip(),
            tables=tables,
            has_tables=len(tables) > 0,
        )
        pages.append(extracted_page)
        all_text_parts.append(page_text.strip())

    metadata = {
        "title": "",
        "author": "",
        "creation_date": "",
        "page_count": len(document.pages),
        "filename": pdf_path.name,
        "extraction_method": "document_ai",
    }

    doc_result = ExtractedDocument(
        source_path=str(pdf_path),
        filename=pdf_path.name,
        page_count=len(pages),
        pages=pages,
        full_text="\n\n".join(all_text_parts),
        metadata=metadata,
    )

    logger.info(
        f"Document AI extracted {pdf_path.name}: {doc_result.page_count} pages, "
        f"{sum(1 for p in pages if p.has_tables)} pages with tables"
    )
    return doc_result


def _filter_headers_footers(text: str) -> str:
    """
    Remove likely header/footer lines from extracted text.

    Real estate PDFs often have headers like 'CONFIDENTIAL - Property Report'
    and footers with page numbers that add noise to embeddings.
    """
    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Skip common header/footer patterns
        if not stripped:
            continue
        if stripped.startswith("CONFIDENTIAL"):
            continue
        if stripped.startswith("Page ") and "|" in stripped:
            continue
        if "Generated by" in stripped:
            continue
        filtered.append(line)
    return "\n".join(filtered)


def _extract_tables_pymupdf(page) -> list[list[list[str]]]:
    """
    Extract tables from a PDF page using PyMuPDF.

    Uses PyMuPDF's built-in table detection to find and extract
    structured tabular data, preserving row/column layout.
    """
    tables = []
    try:
        found_tables = page.find_tables()
        for table in found_tables:
            extracted = table.extract()
            if extracted and len(extracted) > 1:  # at least header + 1 row
                cleaned = []
                for row in extracted:
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    cleaned.append(cleaned_row)
                tables.append(cleaned)
    except Exception as e:
        logger.debug(f"Table extraction failed on page: {e}")

    return tables


# ──────────────────────────────────────────────────────────────────────
# Document AI helper functions
# ──────────────────────────────────────────────────────────────────────

def _get_layout_text(layout, full_text: str) -> str:
    """Extract text from a Document AI layout element using text anchors."""
    result = ""
    if not layout.text_anchor or not layout.text_anchor.text_segments:
        return result
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index)
        end = int(segment.end_index)
        result += full_text[start:end]
    return result.strip()


def _get_page_text(page, full_text: str) -> str:
    """Get all text for a Document AI page from its paragraphs."""
    parts = []
    for paragraph in page.paragraphs:
        text = _get_layout_text(paragraph.layout, full_text)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_table_from_docai(table, full_text: str) -> list[list[str]]:
    """Convert a Document AI table into a list of rows (list of cell strings)."""
    rows = []
    for header_row in table.header_rows:
        cells = [_get_layout_text(cell.layout, full_text) for cell in header_row.cells]
        rows.append(cells)
    for body_row in table.body_rows:
        cells = [_get_layout_text(cell.layout, full_text) for cell in body_row.cells]
        rows.append(cells)
    return rows if len(rows) > 1 else []
