# Real Estate Property Valuation RAG Prototype

A production-quality Retrieval-Augmented Generation (RAG) prototype that solves a real business problem: **parsing proprietary real estate data trapped in unstructured PDF formats that standard RAG tools can't handle.**

Built for the Google Cloud AI Practice, this prototype demonstrates a custom data-cleaning pipeline, hybrid search with metadata filtering, and hallucination-safe generation — the kind of solution a Customer Engineer would build for an enterprise real estate customer.

## The Problem

Large real estate companies have decades of property data locked in siloed, unstructured PDFs: appraisal reports, MLS listings, market analyses, and comparable sales documents. Each document type has a different format, mixes narrative text with tables, and contains structured fields (prices, addresses, bedroom counts) that users need to query precisely.

**Why standard RAG fails:**
- LangChain's PyPDFLoader dumps raw text, losing table structure and comparable sales data
- Fixed-size chunking splits property details across chunks, causing incomplete retrieval
- Pure vector search can't handle structured queries like "3-bedroom homes under $500K in Austin"

## The Solution

A custom RAG pipeline with three key innovations:

1. **Type-specific parsing** — Documents are classified (appraisal, listing, market analysis, comp sales), then parsed by specialized extractors that preserve table structure and extract queryable fields
2. **Hybrid search** — Metadata filtering (bedrooms, price, neighborhood) narrows results BEFORE vector similarity search, then results are reranked by combined score
3. **Hallucination prevention** — The LLM is constrained to retrieved context with source citations, and post-processing verifies that dollar amounts in responses actually appear in the source documents

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER QUERY                        │
│  "3-bedroom homes in Downtown Austin under $600K"   │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Query Classification   │  → Determines prompt template
        │   + Filter Extraction    │  → bedrooms=3, neighborhood=Downtown Austin
        └────────────┬────────────┘     max_price=$600K
                     │
        ┌────────────▼────────────┐
        │    HYBRID SEARCH         │
        │  1. Metadata filtering   │  → ChromaDB where clause
        │  2. Vector similarity    │  → Cosine similarity on filtered set
        │  3. Reranking            │  → Boost metadata matches
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    LLM GENERATION        │
        │  Gemini Pro / GPT-4     │  → Query-specific prompt template
        │  + Source citations      │  → [Source: filename] for each fact
        │  + Number verification   │  → Post-process hallucination check
        └─────────────────────────┘
```

### Ingestion Pipeline

```
PDF Documents → Extract (PyMuPDF) → Classify → Custom Parse → Semantic Chunk → Embed → ChromaDB
                     │                  │            │              │            │
              Document AI          Keyword +     Type-specific   Section-based  Vertex AI /
              (production)        structural     field extraction  with overlap  sentence-transformers
                                  analysis
```

## Project Structure

```
real-estate-rag-prototype/
├── README.md                    # This file
├── requirements.txt             # All dependencies
├── config.py                    # Configuration (backends, API keys, tuning)
├── ingest.py                    # Full ingestion pipeline runner
├── query.py                     # CLI query interface
├── demo.py                      # Pre-scripted demo for interview
├── data/
│   ├── generate_sample_pdfs.py  # Creates realistic sample PDFs
│   └── pdfs/                    # Generated sample PDFs
├── pipeline/
│   ├── pdf_extractor.py         # PDF text and table extraction
│   ├── document_classifier.py   # Classify document type
│   ├── custom_parsers.py        # Type-specific structured extraction
│   └── chunker.py               # Semantic chunking with metadata
├── retrieval/
│   ├── embeddings.py            # Embedding model (Vertex AI / local)
│   ├── vector_store.py          # ChromaDB wrapper
│   └── search.py                # Hybrid search with reranking
├── chatbot/
│   ├── llm_client.py            # Gemini / OpenAI client
│   ├── prompt_templates.py      # Query-specific prompt templates
│   └── chatbot.py               # Main chatbot orchestrator
└── demo/
    ├── cli_demo.py              # Terminal demo with rich formatting
    └── streamlit_app.py         # Web UI demo
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key (for local demo)

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or for Google Cloud (production path):
```bash
export USE_VERTEX_AI=true
export USE_GEMINI=true
export GCP_PROJECT_ID="your-project-id"
gcloud auth application-default login
```

### 3. Run the ingestion pipeline

```bash
python ingest.py
```

This generates sample PDFs, extracts and parses them, creates embeddings, and stores everything in ChromaDB. Takes about 30-60 seconds.

### 4. Run the demo

```bash
# Pre-scripted demo (recommended for interview)
python demo.py

# Interactive chat mode
python demo.py --interactive

# Single query
python query.py "What is the estimated value of 742 Evergreen Terrace?"

# Web UI
streamlit run demo/streamlit_app.py
```

## Google Cloud Production Path

The prototype runs locally for demo reliability, but every component has a Google Cloud production counterpart:

| Prototype (Local) | Production (Google Cloud) |
|---|---|
| PyMuPDF text extraction | **Document AI** — OCR, table extraction, entity detection |
| Keyword-based classification | **Gemini** fine-tuned classifier or Document AI custom classification |
| sentence-transformers embeddings | **Vertex AI Embeddings** (text-embedding-004) |
| ChromaDB | **Vertex AI Vector Search** — billion-scale, managed, filtered search |
| OpenAI GPT-4o-mini | **Gemini Pro** via Vertex AI — enterprise SLAs, data residency |

The Google Cloud code paths are included in the source (see comments in each file) and can be activated by setting environment variables. The architecture is designed for a clean migration from prototype to production.

## Key Technical Decisions

**Why custom parsing over standard RAG:** Standard tools like LangChain's PyPDFLoader dump raw text without preserving table structure. For real estate, losing table structure means losing comparable sales data, which is critical for valuations.

**Why hybrid search:** Pure vector search can't handle "3 bedrooms under $500K in Austin." We need metadata filtering first, then semantic search for relevance.

**Why semantic chunking:** Fixed-size chunking can split a property's details across two chunks. Semantic chunking keeps each property's data together as a meaningful unit.

**Hallucination prevention:** The prompt constrains the LLM to retrieved context only, requires source citations, and post-processing verifies that dollar amounts in responses actually appear in the source documents.

## Demo Queries

The pre-scripted demo (`python demo.py`) runs these queries, each showcasing a different capability:

1. **Simple valuation** — "What is the estimated value of 742 Evergreen Terrace?" → Basic RAG retrieval
2. **Structured filtering** — "3-bedroom properties in Downtown Austin under $600K" → Hybrid search
3. **Comparison** — "Compare 742 Evergreen Terrace with similar homes" → Multi-document retrieval
4. **Market analysis** — "What are the market trends in Austin?" → Cross-document synthesis
5. **Edge case** — "Value at 999 Nonexistent Street, Miami?" → Hallucination prevention
