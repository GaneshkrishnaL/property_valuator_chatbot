"""
Full ingestion pipeline: Generate PDFs → Extract → Classify → Parse → Chunk → Embed → Store.

This is the one-command setup script. Run it before the demo to populate
the vector store with sample data.

Usage:
    python ingest.py
"""

import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import config
from data.generate_sample_pdfs import generate_all_pdfs
from pipeline.pdf_extractor import extract_pdf
from pipeline.document_classifier import classify_document
from pipeline.custom_parsers import get_parser
from pipeline.chunker import chunk_document, DocumentChunk
from retrieval.embeddings import get_embedding_model
from retrieval.vector_store import get_vector_store

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


def run_ingestion() -> None:
    """Run the complete ingestion pipeline with progress tracking."""
    start_time = time.time()

    console.print("\n[bold cyan]Real Estate RAG — Ingestion Pipeline[/bold cyan]\n")

    # ── Step 1: Generate sample PDFs ──
    console.print("[bold]Step 1/6:[/bold] Generating sample PDFs...")
    pdf_paths = generate_all_pdfs()
    console.print(f"  Generated [green]{len(pdf_paths)}[/green] PDFs in {config.PDF_DIR}\n")

    # ── Step 2: Extract text from PDFs ──
    console.print("[bold]Step 2/6:[/bold] Extracting text and tables from PDFs...")
    extracted_docs = []
    for path in pdf_paths:
        if path.exists():
            doc = extract_pdf(path)
            extracted_docs.append(doc)
            console.print(f"  [dim]{doc.filename}: {doc.page_count} pages, "
                         f"{sum(1 for p in doc.pages if p.has_tables)} with tables[/dim]")
    console.print(f"  Extracted [green]{len(extracted_docs)}[/green] documents\n")

    # ── Step 3: Classify documents ──
    console.print("[bold]Step 3/6:[/bold] Classifying document types...")
    classifications = []
    for doc in extracted_docs:
        result = classify_document(doc)
        classifications.append(result)
        console.print(f"  [dim]{doc.filename} → {result.document_type.value} "
                     f"(confidence: {result.confidence:.2f})[/dim]")
    console.print()

    # ── Step 4: Parse with custom parsers ──
    console.print("[bold]Step 4/6:[/bold] Running type-specific parsers...")
    all_chunks: list[DocumentChunk] = []

    for doc, classification in zip(extracted_docs, classifications):
        parser = get_parser(classification.document_type.value)
        if parser:
            parsed = parser.parse(doc)
            chunks = chunk_document(
                filename=doc.filename,
                document_type=classification.document_type.value,
                parsed_data=parsed,
            )
            all_chunks.extend(chunks)
            fields = parsed.get("structured_fields", {})
            field_count = sum(1 for v in fields.values() if v)
            console.print(f"  [dim]{doc.filename}: {field_count} fields extracted, "
                         f"{len(chunks)} chunks created[/dim]")
        else:
            console.print(f"  [yellow]No parser for {classification.document_type.value}, "
                         f"using raw text for {doc.filename}[/yellow]")
            # Fallback: create a single chunk from raw text
            from pipeline.chunker import DocumentChunk
            import hashlib
            chunk = DocumentChunk(
                chunk_id=hashlib.sha256(doc.filename.encode()).hexdigest()[:16],
                text=doc.full_text[:2000],
                document_source=doc.filename,
                document_type=classification.document_type.value,
            )
            all_chunks.append(chunk)

    console.print(f"  Total chunks: [green]{len(all_chunks)}[/green]\n")

    # ── Step 5: Generate embeddings ──
    console.print("[bold]Step 5/6:[/bold] Generating embeddings...")
    embedding_model = get_embedding_model()

    # Batch embed all chunks
    texts = [c.text for c in all_chunks]
    with console.status("[bold green]Embedding chunks..."):
        embeddings = embedding_model.embed_documents(texts)
    console.print(f"  Generated [green]{len(embeddings)}[/green] embeddings "
                 f"({embedding_model.dimension} dimensions)\n")

    # ── Step 6: Store in vector database ──
    backend = "Firestore" if config.USE_FIRESTORE else "ChromaDB"
    console.print(f"[bold]Step 6/6:[/bold] Storing in {backend}...")
    vector_store = get_vector_store()
    vector_store.reset()  # Fresh start
    vector_store.add_chunks(all_chunks, embeddings)
    console.print(f"  Stored [green]{vector_store.count}[/green] chunks in {backend}\n")

    # ── Summary ──
    elapsed = time.time() - start_time
    console.print("[bold cyan]Ingestion Complete![/bold cyan]")
    console.print(f"  PDFs generated:    {len(pdf_paths)}")
    console.print(f"  Documents parsed:  {len(extracted_docs)}")
    console.print(f"  Chunks created:    {len(all_chunks)}")
    console.print(f"  Embeddings stored: {vector_store.count}")
    console.print(f"  Total time:        {elapsed:.1f}s")
    console.print()


if __name__ == "__main__":
    run_ingestion()
