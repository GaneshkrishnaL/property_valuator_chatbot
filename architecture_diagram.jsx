import { useState } from "react";

const COLORS = {
  bg: "#0f172a",
  card: "#1e293b",
  cardHover: "#334155",
  accent: "#3b82f6",
  accentLight: "#60a5fa",
  green: "#22c55e",
  yellow: "#eab308",
  orange: "#f97316",
  purple: "#a855f7",
  pink: "#ec4899",
  text: "#f8fafc",
  textDim: "#94a3b8",
  border: "#475569",
  arrow: "#60a5fa",
};

const files = {
  "generate_sample_pdfs.py": {
    phase: "data",
    title: "PDF Generator",
    shortDesc: "Creates 11 realistic real estate PDFs",
    fullDesc: `This script generates fake-but-realistic PDF documents that simulate what a real estate company would have. It creates 4 types of documents:

• Appraisal Reports (3 PDFs) — These are like official valuation documents. They have a mix of narrative text ("The property is in good condition...") AND tables (comparable sales with prices, square footage). This mix is exactly what makes them hard for standard RAG tools to parse.

• Listing Sheets (6 PDFs) — MLS-style property listings with features tables, agent remarks, and pricing. Each one has a different format, just like real MLS data.

• Market Analysis (1 PDF) — Neighborhood statistics with tables showing average prices, days on market, and trends across 5 Austin neighborhoods.

• Comparable Sales Report (1 PDF) — A table of 12 recent sales with addresses, prices, dates, and price-per-sqft.

WHY IT'S BUILT THIS WAY: The PDFs are intentionally messy — tables with tight columns, headers/footers that look like content, narrative text mixed with numbers. This demonstrates to the interviewer WHY a custom pipeline is needed. Standard tools like LangChain's PyPDFLoader would just dump all this as flat text and lose the table structure.

KEY LIBRARY: Uses fpdf2 to generate PDFs programmatically. The RealEstatePDF class adds headers, footers, and table formatting.`,
  },
  "pdf_extractor.py": {
    phase: "pipeline",
    title: "PDF Extractor",
    shortDesc: "Extracts text + tables from PDFs",
    fullDesc: `This is the first step of the pipeline. It opens each PDF and pulls out two things: the raw text AND any tables it finds.

HOW IT WORKS:
1. Opens the PDF using PyMuPDF (the "fitz" library)
2. For each page, extracts the raw text
3. Filters out headers/footers (lines with "CONFIDENTIAL", "Page X |", etc.) so they don't pollute the embeddings
4. Uses PyMuPDF's built-in table detection to find and extract tabular data, preserving the row/column structure
5. Collects document metadata (filename, page count, creation date)

WHAT IT OUTPUTS: An ExtractedDocument object containing:
- pages: list of ExtractedPage objects, each with text + tables
- full_text: all pages concatenated
- metadata: document-level info

WHY NOT JUST USE LANGCHAIN? LangChain's PyPDFLoader gives you flat text — it doesn't distinguish tables from paragraphs. When you have a comparable sales table with 10 rows of data, LangChain gives you "750 Evergreen Terrace $535,000 2024-08-15 2,380" as one long string. Our extractor keeps the table structure intact so the parser can extract each field.

PRODUCTION NOTE: The file includes commented-out Google Document AI code showing how you'd swap PyMuPDF for Document AI in production (OCR for scanned docs, ML-based table extraction, entity detection).`,
  },
  "document_classifier.py": {
    phase: "pipeline",
    title: "Document Classifier",
    shortDesc: "Identifies document type (appraisal, listing, etc.)",
    fullDesc: `After extraction, we need to know WHAT TYPE of document we're dealing with, because each type needs a different parser.

HOW IT WORKS:
1. Takes the full extracted text
2. Runs a set of weighted regex patterns against it
3. Each document type has specific "signal" patterns:
   - APPRAISAL: "appraisal report" (0.35 weight), "appraiser" (0.20), "assessed value" (0.15)
   - LISTING: "property listing" (0.30), "MLS#" (0.25), "list price" (0.20)
   - MARKET_ANALYSIS: "market analysis" (0.35), "year-over-year" (0.15)
   - COMPARABLE_SALES: "comparable sales report" (0.35), "$/sq ft" (0.15)
4. Adds structural bonuses (2+ pages → likely appraisal, lots of tables → likely comp sales)
5. The type with the highest score wins

WHAT IT OUTPUTS: A ClassificationResult with:
- document_type: APPRAISAL, LISTING, MARKET_ANALYSIS, COMPARABLE_SALES, or UNKNOWN
- confidence: 0.0 to 1.0
- reasoning: human-readable explanation

WHY THIS MATTERS: The classification determines which PARSER handles the document next. An appraisal parser knows to look for "assessed value" and "appraiser notes", while a listing parser looks for "MLS#" and "days on market". Without classification, you'd need one giant parser that handles everything — messy and fragile.`,
  },
  "custom_parsers.py": {
    phase: "pipeline",
    title: "Custom Parsers",
    shortDesc: "Extracts structured fields per document type",
    fullDesc: `THIS IS THE MOST IMPORTANT FILE. It's the whole reason the prototype exists — it solves the "proprietary data trapped in unstructured PDFs" problem.

There are 4 parser classes, one per document type:

APPRAISAL PARSER extracts:
- property_address, square_footage, bedrooms, bathrooms, lot_size, year_built
- assessed_value, estimated_value (dollar amounts via regex)
- comparable_sales table (iterates through extracted tables, matches columns by header names)
- Splits text into sections (Introduction, Assessment, Conclusion) for semantic chunking

LISTING PARSER extracts:
- listing_price, MLS number, days_on_market, agent_info
- Property features from the features table
- Agent remarks (narrative text section)

MARKET ANALYSIS PARSER extracts:
- Per-neighborhood stats from tables (avg_price, median_price, inventory, YoY change)
- Splits by neighborhood sections for targeted retrieval

COMPARABLE SALES PARSER extracts:
- Individual sales from tables (address, price, date, sqft, price_per_sqft)
- Computes aggregate stats (average, min, max prices)

HOW PARSING WORKS (example):
When the AppraisalParser sees "built in 2018", the regex r"built\\s+in\\s+(\\d{4})" captures "2018" and stores it as year_built=2018. This becomes METADATA in the vector store, so when someone asks "homes built after 2015", we can filter on that field BEFORE doing expensive vector search.

EACH PARSER OUTPUTS TWO THINGS:
1. structured_fields → become searchable metadata (bedrooms=4, price=525000)
2. text_sections → become the text content that gets embedded`,
  },
  "chunker.py": {
    phase: "pipeline",
    title: "Semantic Chunker",
    shortDesc: "Splits documents into meaningful chunks with metadata",
    fullDesc: `After parsing, we need to break documents into chunks small enough for the embedding model (500-800 tokens each) but smart enough to keep related info together.

HOW SEMANTIC CHUNKING WORKS:
1. Takes the text_sections from the parser (e.g., "Subject Property Info", "Appraiser's Assessment", "Valuation Conclusion")
2. Each section becomes one chunk IF it fits (under ~1500 chars)
3. If a section is too long, splits on PARAGRAPH BOUNDARIES (not character count)
4. Adds OVERLAP: the last paragraph of chunk N is repeated at the start of chunk N+1, so context isn't lost at boundaries

WHY NOT FIXED-SIZE CHUNKING?
Fixed-size (e.g., every 500 chars) might split "Property: 742 Evergreen Terrace" into chunk 1 and "Price: $525,000" into chunk 2. Now when someone asks about that property's price, neither chunk alone has the full answer. Semantic chunking keeps the whole "Valuation Conclusion" section together.

THE DocumentChunk MODEL (Pydantic):
Each chunk has:
- chunk_id: deterministic hash for deduplication
- text: the actual content for embedding
- document_source: which PDF it came from
- document_type: APPRAISAL, LISTING, etc.
- Metadata fields: property_address, neighborhood, bedrooms, bathrooms, square_footage, price, year_built

These metadata fields are THE KEY to hybrid search. They come from the parser's structured_fields and get stored alongside the embedding in ChromaDB.

BONUS: For property documents, it creates an extra "structured summary" chunk that's just the key facts in a clean format. This ensures queries like "tell me about 742 Evergreen" find a concise summary.`,
  },
  "embeddings.py": {
    phase: "retrieval",
    title: "Embedding Models",
    shortDesc: "Converts text to vectors (local or Vertex AI)",
    fullDesc: `Embeddings convert text into numerical vectors (arrays of numbers) that capture meaning. Similar texts get similar vectors, which is how semantic search works.

TWO BACKENDS:

1. LOCAL (demo mode) — sentence-transformers/all-MiniLM-L6-v2
   - 384-dimensional vectors
   - Runs on CPU, no API key needed
   - Small and fast (~80MB model)
   - Perfect for live demo — no network dependency

2. VERTEX AI (production) — text-embedding-004
   - 768-dimensional vectors
   - Google's latest embedding model
   - Supports "task type" hints: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for searching
   - These hints optimize the embedding for search scenarios

HOW IT'S USED:
- During INGESTION: embed_documents() converts all chunks into vectors
- During SEARCH: embed_query() converts the user's question into a vector
- The search finds chunks whose vectors are closest to the query vector (cosine similarity)

THE PROTOCOL PATTERN:
Both models implement the same EmbeddingModel protocol (embed_documents, embed_query, dimension). This means the rest of the code doesn't care which model is running — you switch with one config flag.

FACTORY FUNCTION:
get_embedding_model() checks config.USE_VERTEX_AI and returns the right one. Clean separation.`,
  },
  "vector_store.py": {
    phase: "retrieval",
    title: "Vector Store (ChromaDB)",
    shortDesc: "Stores and searches embeddings with metadata",
    fullDesc: `ChromaDB is an open-source vector database. Think of it as a specialized database optimized for "find me things similar to this."

WHAT IT STORES (per chunk):
- The embedding vector (384 or 768 numbers)
- The original text
- Metadata dict (document_type, property_address, bedrooms, price, etc.)

KEY OPERATIONS:

add_chunks(): Stores chunks with their embeddings and metadata. Called during ingestion.

search(): The core operation. Takes a query embedding and returns the most similar chunks.
  - Uses COSINE SIMILARITY: measures the angle between two vectors. Score of 1.0 = identical, 0.0 = completely unrelated.
  - Supports a "where" parameter for METADATA FILTERING. This is crucial — it narrows the search space before doing vector math.
  - Example: where={"bedrooms": 3, "neighborhood": "Downtown Austin"} only searches chunks that match those exact values.

reset(): Deletes everything and recreates the collection. Used during re-ingestion.

WHY CHROMADB FOR THE PROTOTYPE?
- Runs locally, no cloud setup needed
- Persists to disk (survives restarts)
- Built-in metadata filtering
- Easy to understand and demo

PRODUCTION NOTE: In production, Vertex AI Vector Search handles billions of vectors with managed scaling. The migration is straightforward — same embeddings, same metadata, just swap the storage backend.`,
  },
  "search.py": {
    phase: "retrieval",
    title: "Hybrid Search",
    shortDesc: "Metadata filter → vector search → rerank",
    fullDesc: `This is WHERE THE MAGIC HAPPENS. It's the component that makes this prototype better than standard RAG.

THREE-STAGE SEARCH:

STAGE 1 — EXTRACT FILTERS from the query:
"3-bedroom homes in Downtown Austin under $500K"
→ bedrooms=3, neighborhood="Downtown Austin", max_price=500000

Uses regex to parse:
- Bedroom count: r"(\\d+)\\s*bedroom" → bedrooms=3
- Price caps: r"under\\s*\\$(\\d+)k?" → max_price=500000
- Neighborhoods: checks against known list
- Document type hints: "market trends" → MARKET_ANALYSIS

STAGE 2 — METADATA FILTERING + VECTOR SEARCH:
Builds a ChromaDB "where" clause from the extracted filters. This narrows the search to only chunks that match the hard constraints BEFORE doing expensive vector similarity. Then cosine similarity ranks the filtered results.

If filtering returns nothing (too restrictive), it automatically falls back to unfiltered search.

STAGE 3 — RERANKING:
After retrieval, results are re-scored by combining:
- Base vector similarity score
- Metadata match bonuses (+0.15 for matching address, +0.08 for neighborhood, +0.05 for doc type and bedrooms)

WHY HYBRID SEARCH?
Pure vector search for "3-bedroom homes under $500K" might return a 5-bedroom $800K home that's semantically similar (it talks about bedrooms and prices). Metadata filtering ensures hard constraints are respected. The reranking then sorts by relevance within the filtered set.`,
  },
  "llm_client.py": {
    phase: "chatbot",
    title: "LLM Client",
    shortDesc: "Calls Gemini or OpenAI for text generation",
    fullDesc: `Abstraction layer for calling the language model. Same pattern as embeddings — two backends, one interface.

TWO BACKENDS:

1. OPENAI (demo mode) — GPT-4o-mini
   - Fast, cheap, reliable
   - You provide your API key via OPENAI_API_KEY env var
   - Uses temperature=0.3 (low randomness for factual answers)
   - Supports both regular and streaming responses

2. GEMINI (production) — gemini-1.5-pro via Vertex AI
   - Google's model, shown to the interviewer as the production path
   - Enterprise SLAs, data residency, IAM integration
   - Also supports streaming

BOTH SUPPORT TWO METHODS:
- generate(): Returns the complete response (used in demo.py)
- generate_stream(): Yields tokens one at a time (used for interactive mode — looks impressive during live demo)

THE PROTOCOL PATTERN:
LLMClient protocol defines generate() and generate_stream(). Both OpenAIClient and GeminiClient implement it. The factory function get_llm_client() returns whichever is configured.

TEMPERATURE=0.3: We use low temperature because this is a factual Q&A system. We don't want creative responses — we want accurate data retrieval and citation.`,
  },
  "prompt_templates.py": {
    phase: "chatbot",
    title: "Prompt Templates",
    shortDesc: "Query-specific prompts with hallucination prevention",
    fullDesc: `Different types of questions need different instructions for the LLM. This file has 4 templates plus a shared system prompt.

SYSTEM PROMPT (shared across all queries):
Sets the role ("property valuation assistant"), and enforces critical rules:
1. ONLY use provided context — never general knowledge
2. ALWAYS cite sources with [Source: filename]
3. Say "I don't have sufficient data" if context is insufficient
4. Never invent property values
5. Distinguish between assessed/listing/sale/estimated values

QUERY-SPECIFIC TEMPLATES:

VALUATION: "Look for explicitly stated values first. Report the number from the Valuation Conclusion section. Do NOT compute your own estimate."
→ This prevents the LLM from doing its own math when the document already contains the answer.

COMPARISON: "Compare specific properties side by side. List key attributes. Highlight differences."
→ Guides the LLM to produce a structured comparison table.

MARKET ANALYSIS: "Focus on trends, statistics, neighborhood data. Include specific numbers."
→ Tells the LLM to pull all the quantitative data.

GENERAL: "Present whatever matching properties you find, even if there is only one."
→ Prevents the "insufficient data" response when there's actually useful info.

QUERY CLASSIFICATION:
classify_query() looks at keywords to pick the right template:
- "value", "worth", "appraisal" → VALUATION
- "compare", "vs", "similar" → COMPARISON
- "market", "trend", "average" → MARKET_ANALYSIS
- Everything else → GENERAL`,
  },
  "chatbot.py": {
    phase: "chatbot",
    title: "Chatbot Orchestrator",
    shortDesc: "Ties everything together: retrieve → generate → verify",
    fullDesc: `This is the MAIN BRAIN of the system. It coordinates the entire flow from user question to final answer.

THE ask() METHOD — step by step:

1. SEARCH: Calls hybrid_search() with the RAW query (not enriched with history — we fixed that bug). This extracts filters, does metadata-filtered vector search, and reranks.

2. CLASSIFY: Determines query type (valuation, comparison, market, general) to pick the right prompt template.

3. CHECK CONTEXT: If search returned nothing, immediately returns "I don't have sufficient data" — no LLM call needed.

4. BUILD PROMPT: Formats the retrieved documents into the prompt template. If conversation history exists, appends it to the LLM prompt (NOT the search query) so the LLM can handle follow-ups.

5. CALL LLM: Sends system prompt + user prompt to OpenAI/Gemini.

6. VERIFY NUMBERS: Post-processes the answer to check if dollar amounts appear in the retrieved context. Amounts close to context values (within 20%) are considered "derived" (like computed averages). Amounts far from any context value get flagged as potential hallucinations.

7. EXTRACT SOURCES: Pulls [Source: filename] citations from the answer, plus adds top retrieved documents.

8. ASSESS CONFIDENCE: Based on retrieval scores — high (top score > 0.7), medium (> 0.5), low, or insufficient_data.

CONVERSATION HISTORY:
The ConversationHistory class stores past Q&A turns. It's used ONLY in the LLM prompt (step 4), never in search (step 1). This was the bug we fixed — history was contaminating filter extraction.

STREAMING:
ask_stream() yields tokens for real-time display in interactive mode.`,
  },
  "cli_demo.py": {
    phase: "demo",
    title: "CLI Demo",
    shortDesc: "Rich terminal output for interview screen share",
    fullDesc: `Beautiful terminal UI using the "rich" library. This is what appears on screen during your interview.

WHAT IT SHOWS:
- Header box with project name
- Each query numbered (Query 1/5, 2/5, etc.)
- Retrieved Documents TABLE: source filename, doc type, relevance score (color-coded green/yellow/red), text preview
- Answer in a bordered PANEL with confidence badge (HIGH/MEDIUM/LOW/INSUFFICIENT DATA)
- Source citations below the answer
- "Press Enter for next query..." between queries for presentation pacing

TWO MODES:
1. run_demo_queries(): Runs the 5 pre-scripted queries from demo.py — this is for the interview presentation
2. run_interactive(): Free-form chat where you type questions — good for Q&A when the interviewer wants to try their own queries

WHY RICH LIBRARY?
During a screen share, plain print() output looks amateur. Rich gives you colored tables, bordered panels, spinners, and formatted text that looks professional and polished.`,
  },
  "streamlit_app.py": {
    phase: "demo",
    title: "Streamlit Web UI",
    shortDesc: "Web-based chat interface with sidebar",
    fullDesc: `A web application that provides a visual chat interface. Run it with: streamlit run demo/streamlit_app.py

LAYOUT:
- LEFT SIDEBAR: Architecture description, example query buttons
- MAIN AREA: Chat interface with message history

FEATURES:
- Click example queries in the sidebar to run them
- Expandable "Retrieved Documents" section showing what was found
- Expandable "Extracted Filters" showing the metadata filters parsed from your query
- Color-coded confidence badges
- Source citation badges
- Full conversation history (persists across queries via Streamlit session state)

THE @st.cache_resource DECORATOR:
The chatbot is initialized once and cached across Streamlit reruns. Without this, every time you type a query, Streamlit would reload the embedding model (3+ seconds). With caching, only the first load is slow.

WHEN TO USE IT:
The Streamlit app is more impressive visually, but the CLI demo is more reliable for a live interview (no browser, no port issues). Use Streamlit if the interviewer asks to see a web interface, or if you have time at the end of your presentation.`,
  },
  "config.py": {
    phase: "config",
    title: "Configuration",
    shortDesc: "Backend selection, paths, and tuning parameters",
    fullDesc: `Central configuration file. All settings in one place — no magic numbers scattered across the codebase.

KEY SETTINGS:

BACKEND FLAGS:
- USE_VERTEX_AI: switches embedding model (local vs Google Cloud)
- USE_GEMINI: switches LLM (OpenAI vs Gemini)
Both default to False (local mode) for demo reliability.

GOOGLE CLOUD SETTINGS:
- GCP_PROJECT_ID, GCP_LOCATION, model names
- DOCUMENT_AI_PROCESSOR_ID for the production extraction path

RETRIEVAL TUNING:
- CHROMA_COLLECTION_NAME: name of the vector DB collection
- TOP_K_RESULTS = 5: how many chunks to retrieve per query
- CHUNK_SIZE_TOKENS = 600: target chunk size
- CHUNK_OVERLAP_TOKENS = 80: overlap between chunks
- SIMILARITY_THRESHOLD = 0.3: minimum score to include a result

All settings can be overridden via environment variables, so you can switch modes without editing code:
  export USE_GEMINI=true
  export GCP_PROJECT_ID=my-project`,
  },
  "ingest.py": {
    phase: "entry",
    title: "Ingestion Pipeline",
    shortDesc: "One command to set up everything",
    fullDesc: `The setup script. Run "python ingest.py" and it does everything:

STEP 1: Generate sample PDFs → Creates 11 PDFs in data/pdfs/
STEP 2: Extract text and tables → PyMuPDF processes each PDF
STEP 3: Classify documents → Keyword matching determines type
STEP 4: Parse with custom parsers → Type-specific field extraction
STEP 5: Generate embeddings → Converts 44 chunks into vectors
STEP 6: Store in ChromaDB → Everything indexed and searchable

Prints progress at each step with counts (documents processed, chunks created, fields extracted). Takes about 6-10 seconds total.

You run this ONCE before the demo. The ChromaDB data persists on disk, so you don't need to re-run it unless you change the sample data.`,
  },
  "demo.py": {
    phase: "entry",
    title: "Demo Runner",
    shortDesc: "Pre-scripted queries for the interview",
    fullDesc: `The script you run during the interview: "python demo.py"

Contains 5 pre-scripted queries that each demonstrate a different capability:

1. VALUATION: "What is the estimated value of 742 Evergreen Terrace?"
   → Shows basic RAG: retrieve the right appraisal, cite the stated value

2. FILTERING: "What about 3-bedroom homes in Downtown Austin under $600K?"
   → Shows hybrid search: metadata filters (bedrooms=3, neighborhood, max_price) narrow results before vector search

3. COMPARISON: "Compare 742 Evergreen with similar homes"
   → Shows multi-document retrieval: pulls from appraisal + listing + comp sales

4. MARKET ANALYSIS: "What are the market trends in Austin?"
   → Shows cross-document synthesis: aggregates data from the market analysis report

5. EDGE CASE: "Value at 999 Nonexistent Street, Miami?"
   → Shows hallucination prevention: says "I don't have sufficient data" instead of making something up

Also supports --interactive mode for free-form Q&A.`,
  },
};

const phases = [
  {
    id: "data",
    label: "1. DATA GENERATION",
    color: COLORS.green,
    description: "Create realistic sample PDFs",
    files: ["generate_sample_pdfs.py"],
  },
  {
    id: "pipeline",
    label: "2. CUSTOM PIPELINE",
    color: COLORS.orange,
    description: "Extract → Classify → Parse → Chunk",
    files: ["pdf_extractor.py", "document_classifier.py", "custom_parsers.py", "chunker.py"],
  },
  {
    id: "retrieval",
    label: "3. RETRIEVAL",
    color: COLORS.accent,
    description: "Embed → Store → Hybrid Search",
    files: ["embeddings.py", "vector_store.py", "search.py"],
  },
  {
    id: "chatbot",
    label: "4. CHATBOT",
    color: COLORS.purple,
    description: "LLM → Prompt → Orchestrate",
    files: ["llm_client.py", "prompt_templates.py", "chatbot.py"],
  },
  {
    id: "demo",
    label: "5. DEMO UI",
    color: COLORS.pink,
    description: "CLI + Streamlit interfaces",
    files: ["cli_demo.py", "streamlit_app.py"],
  },
  {
    id: "entry",
    label: "6. ENTRY POINTS",
    color: COLORS.yellow,
    description: "ingest.py, demo.py",
    files: ["ingest.py", "demo.py"],
  },
  {
    id: "config",
    label: "CONFIG",
    color: COLORS.textDim,
    description: "Settings & backend selection",
    files: ["config.py"],
  },
];

const Arrow = ({ from, to, label }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "4px 0" }}>
    <div style={{ color: COLORS.arrow, fontSize: "18px", fontWeight: "bold" }}>↓</div>
    {label && (
      <span style={{ color: COLORS.textDim, fontSize: "11px", marginLeft: "8px", fontStyle: "italic" }}>{label}</span>
    )}
  </div>
);

const FlowDiagram = ({ onSelectFile }) => {
  const flowSteps = [
    { icon: "📄", label: "User Query", sub: '"3-bed homes in Downtown Austin under $600K"', color: COLORS.text },
    { icon: "🔍", label: "Extract Filters", sub: "bedrooms=3, neighborhood=Downtown Austin, max_price=$600K", color: COLORS.orange, file: "search.py" },
    { icon: "🗂️", label: "Metadata Filter", sub: "ChromaDB WHERE clause narrows to matching chunks", color: COLORS.accent, file: "vector_store.py" },
    { icon: "📐", label: "Vector Similarity", sub: "Cosine similarity ranks filtered results", color: COLORS.accent, file: "embeddings.py" },
    { icon: "⚖️", label: "Rerank", sub: "Boost results matching more filters", color: COLORS.accent, file: "search.py" },
    { icon: "📝", label: "Build Prompt", sub: "Query-specific template + retrieved context", color: COLORS.purple, file: "prompt_templates.py" },
    { icon: "🤖", label: "LLM Generation", sub: "GPT-4o-mini / Gemini Pro", color: COLORS.purple, file: "llm_client.py" },
    { icon: "✅", label: "Verify & Cite", sub: "Check numbers + add [Source: file] citations", color: COLORS.green, file: "chatbot.py" },
  ];

  return (
    <div style={{ padding: "16px", background: "#0c1222", borderRadius: "12px", marginBottom: "20px" }}>
      <div style={{ color: COLORS.text, fontWeight: "bold", fontSize: "15px", marginBottom: "12px", textAlign: "center" }}>
        Query Flow (what happens when you ask a question)
      </div>
      {flowSteps.map((step, i) => (
        <div key={i}>
          <div
            onClick={() => step.file && onSelectFile(step.file)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "8px 12px",
              borderRadius: "8px",
              border: `1px solid ${step.color}33`,
              background: `${step.color}11`,
              cursor: step.file ? "pointer" : "default",
              transition: "background 0.2s",
            }}
            onMouseEnter={(e) => step.file && (e.currentTarget.style.background = `${step.color}22`)}
            onMouseLeave={(e) => step.file && (e.currentTarget.style.background = `${step.color}11`)}
          >
            <span style={{ fontSize: "20px" }}>{step.icon}</span>
            <div>
              <div style={{ color: step.color, fontWeight: "bold", fontSize: "13px" }}>{step.label}</div>
              <div style={{ color: COLORS.textDim, fontSize: "11px" }}>{step.sub}</div>
            </div>
          </div>
          {i < flowSteps.length - 1 && <Arrow />}
        </div>
      ))}
    </div>
  );
};

const IngestionDiagram = ({ onSelectFile }) => {
  const steps = [
    { icon: "📄", label: "Sample PDFs", sub: "11 realistic real estate documents", color: COLORS.green, file: "generate_sample_pdfs.py" },
    { icon: "📖", label: "Extract", sub: "PyMuPDF: text + tables + metadata", color: COLORS.orange, file: "pdf_extractor.py" },
    { icon: "🏷️", label: "Classify", sub: "Keyword matching → APPRAISAL / LISTING / ...", color: COLORS.orange, file: "document_classifier.py" },
    { icon: "⚙️", label: "Parse", sub: "Type-specific field extraction (address, price, beds...)", color: COLORS.orange, file: "custom_parsers.py" },
    { icon: "✂️", label: "Chunk", sub: "Semantic sections + rich metadata", color: COLORS.orange, file: "chunker.py" },
    { icon: "🔢", label: "Embed", sub: "Text → 384-dim vectors", color: COLORS.accent, file: "embeddings.py" },
    { icon: "💾", label: "Store", sub: "ChromaDB: vectors + metadata", color: COLORS.accent, file: "vector_store.py" },
  ];

  return (
    <div style={{ padding: "16px", background: "#0c1222", borderRadius: "12px", marginBottom: "20px" }}>
      <div style={{ color: COLORS.text, fontWeight: "bold", fontSize: "15px", marginBottom: "12px", textAlign: "center" }}>
        Ingestion Pipeline (python ingest.py)
      </div>
      {steps.map((step, i) => (
        <div key={i}>
          <div
            onClick={() => onSelectFile(step.file)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "8px 12px",
              borderRadius: "8px",
              border: `1px solid ${step.color}33`,
              background: `${step.color}11`,
              cursor: "pointer",
              transition: "background 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = `${step.color}22`)}
            onMouseLeave={(e) => (e.currentTarget.style.background = `${step.color}11`)}
          >
            <span style={{ fontSize: "20px" }}>{step.icon}</span>
            <div>
              <div style={{ color: step.color, fontWeight: "bold", fontSize: "13px" }}>{step.label}</div>
              <div style={{ color: COLORS.textDim, fontSize: "11px" }}>{step.sub}</div>
            </div>
          </div>
          {i < steps.length - 1 && <Arrow />}
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [activeTab, setActiveTab] = useState("ingestion");

  const fileInfo = selectedFile ? files[selectedFile] : null;

  return (
    <div style={{ fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif", background: COLORS.bg, color: COLORS.text, minHeight: "100vh", padding: "20px" }}>
      <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
        <h1 style={{ fontSize: "22px", fontWeight: "bold", marginBottom: "4px" }}>
          Real Estate RAG Prototype — Architecture
        </h1>
        <p style={{ color: COLORS.textDim, fontSize: "13px", marginBottom: "20px" }}>
          Click any component to see a detailed explanation of what it does and how it works.
        </p>

        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
          {/* Left: Diagrams */}
          <div style={{ flex: "1 1 380px", minWidth: "340px" }}>
            {/* Tab switcher */}
            <div style={{ display: "flex", gap: "4px", marginBottom: "12px" }}>
              {[
                { id: "ingestion", label: "Ingestion Flow" },
                { id: "query", label: "Query Flow" },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    flex: 1,
                    padding: "8px",
                    border: `1px solid ${activeTab === tab.id ? COLORS.accent : COLORS.border}`,
                    background: activeTab === tab.id ? `${COLORS.accent}22` : "transparent",
                    color: activeTab === tab.id ? COLORS.accentLight : COLORS.textDim,
                    borderRadius: "8px",
                    cursor: "pointer",
                    fontSize: "13px",
                    fontWeight: activeTab === tab.id ? "bold" : "normal",
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {activeTab === "ingestion" ? (
              <IngestionDiagram onSelectFile={setSelectedFile} />
            ) : (
              <FlowDiagram onSelectFile={setSelectedFile} />
            )}

            {/* File grid */}
            <div style={{ color: COLORS.textDim, fontSize: "12px", fontWeight: "bold", marginBottom: "8px" }}>
              ALL FILES (click to explore)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
              {Object.entries(files).map(([name, info]) => {
                const phase = phases.find((p) => p.id === info.phase);
                const isActive = selectedFile === name;
                return (
                  <button
                    key={name}
                    onClick={() => setSelectedFile(name)}
                    style={{
                      padding: "8px 10px",
                      border: `1px solid ${isActive ? phase?.color || COLORS.border : COLORS.border}`,
                      background: isActive ? `${phase?.color || COLORS.accent}22` : COLORS.card,
                      color: isActive ? phase?.color : COLORS.text,
                      borderRadius: "8px",
                      cursor: "pointer",
                      textAlign: "left",
                      fontSize: "11px",
                      transition: "all 0.2s",
                    }}
                  >
                    <div style={{ fontWeight: "bold", marginBottom: "2px" }}>{name}</div>
                    <div style={{ color: COLORS.textDim, fontSize: "10px" }}>{info.shortDesc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right: Detail panel */}
          <div style={{ flex: "1 1 420px", minWidth: "340px" }}>
            {fileInfo ? (
              <div
                style={{
                  background: COLORS.card,
                  borderRadius: "12px",
                  padding: "20px",
                  border: `1px solid ${COLORS.border}`,
                  position: "sticky",
                  top: "20px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "12px" }}>
                  <div>
                    <div style={{ fontSize: "11px", color: phases.find((p) => p.id === fileInfo.phase)?.color, fontWeight: "bold", textTransform: "uppercase", marginBottom: "4px" }}>
                      {phases.find((p) => p.id === fileInfo.phase)?.label}
                    </div>
                    <h2 style={{ fontSize: "18px", fontWeight: "bold", margin: 0 }}>{fileInfo.title}</h2>
                    <div style={{ color: COLORS.accentLight, fontSize: "13px", fontFamily: "monospace" }}>{selectedFile}</div>
                  </div>
                  <button
                    onClick={() => setSelectedFile(null)}
                    style={{
                      background: "transparent",
                      border: `1px solid ${COLORS.border}`,
                      color: COLORS.textDim,
                      borderRadius: "6px",
                      padding: "4px 8px",
                      cursor: "pointer",
                      fontSize: "12px",
                    }}
                  >
                    Close
                  </button>
                </div>
                <div
                  style={{
                    color: COLORS.textDim,
                    fontSize: "13px",
                    lineHeight: "1.7",
                    whiteSpace: "pre-wrap",
                    maxHeight: "calc(100vh - 160px)",
                    overflowY: "auto",
                  }}
                >
                  {fileInfo.fullDesc}
                </div>
              </div>
            ) : (
              <div
                style={{
                  background: COLORS.card,
                  borderRadius: "12px",
                  padding: "40px 20px",
                  border: `1px solid ${COLORS.border}`,
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "40px", marginBottom: "12px" }}>👈</div>
                <div style={{ color: COLORS.textDim, fontSize: "14px" }}>
                  Click any component in the diagram or file grid to see a detailed explanation
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
