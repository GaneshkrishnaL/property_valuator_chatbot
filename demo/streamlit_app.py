"""
Streamlit web UI for the Property Valuation RAG Chatbot.

Provides a polished web interface showing:
  - Chat input with conversation history
  - Retrieved documents panel with scores
  - Source citations
  - System architecture diagram

Run with: streamlit run demo/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.chatbot import PropertyValuationChatbot
from retrieval.search import extract_query_filters


# ── Page config ──
st.set_page_config(
    page_title="Property Valuation Chatbot",
    page_icon="🏠",
    layout="wide",
)

# ── Custom CSS for clean look ──
st.markdown("""
<style>
.source-badge {
    background: #e8f0fe;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    margin-right: 6px;
    display: inline-block;
    margin-bottom: 4px;
}
.confidence-high { color: #137333; font-weight: bold; }
.confidence-medium { color: #b06000; font-weight: bold; }
.confidence-low { color: #c5221f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Initialize the chatbot (cached across Streamlit reruns)."""
    return PropertyValuationChatbot()


def main():
    # ── Header ──
    st.title("Property Valuation RAG Chatbot")
    st.caption("Custom RAG Pipeline with Hybrid Search | Google Cloud AI Practice Prototype")

    # ── Sidebar: architecture info ──
    with st.sidebar:
        st.header("Architecture")
        st.markdown("""
        **Pipeline:**
        1. PDF Extraction (PyMuPDF / Document AI)
        2. Document Classification
        3. Custom Type-Specific Parsing
        4. Semantic Chunking with Metadata
        5. Embedding (Vertex AI / sentence-transformers)
        6. ChromaDB Vector Store

        **Retrieval:**
        - Metadata filtering (bedrooms, price, area)
        - Vector similarity search
        - Reranking by combined score

        **Generation:**
        - Gemini Pro / GPT-4o-mini
        - Query-specific prompt templates
        - Source citations + hallucination checks
        """)

        st.divider()
        st.subheader("Example Queries")
        examples = [
            "What is the estimated value of 742 Evergreen Terrace, Austin?",
            "What can you tell me about 3-bedroom homes in Downtown Austin under $600,000?",
            "Compare properties in East Austin vs South Lamar",
            "What are the market trends in Austin?",
            "What is the value at 999 Nonexistent Street, Miami?",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.pending_query = ex

    # ── Initialize ──
    chatbot = load_chatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # ── Display chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.caption("Sources:")
                with cols[1]:
                    source_html = "".join(
                        f'<span class="source-badge">{s}</span>' for s in msg["sources"]
                    )
                    st.markdown(source_html, unsafe_allow_html=True)

    # ── Handle input ──
    query = st.chat_input("Ask about Austin real estate...")

    # Check for sidebar example click
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Show extracted filters
        filters = extract_query_filters(query)
        active_filters = {k: v for k, v in filters.model_dump().items() if v is not None}
        if active_filters:
            with st.expander("Extracted Filters", expanded=False):
                st.json(active_filters)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = chatbot.ask(query)

            # Show retrieved documents in expander
            if response.retrieved_documents:
                with st.expander(f"Retrieved {len(response.retrieved_documents)} documents", expanded=False):
                    for i, doc in enumerate(response.retrieved_documents, 1):
                        score_color = "green" if doc.score > 0.6 else "orange" if doc.score > 0.4 else "red"
                        st.markdown(
                            f"**{i}. {doc.document_source}** "
                            f"(:{score_color}[{doc.score:.3f}]) — {doc.document_type}"
                        )
                        st.text(doc.text[:200] + "...")
                        st.divider()

            # Show answer
            st.markdown(response.answer)

            # Confidence and sources
            conf_class = f"confidence-{response.confidence}"
            st.markdown(
                f'<span class="{conf_class}">Confidence: {response.confidence.upper()}</span>',
                unsafe_allow_html=True,
            )

            if response.sources:
                source_html = "".join(
                    f'<span class="source-badge">{s}</span>' for s in response.sources
                )
                st.markdown(source_html, unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })


if __name__ == "__main__":
    main()
