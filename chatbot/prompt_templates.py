"""
Prompt templates for the property valuation chatbot.

ARCHITECTURAL NOTE — Hallucination prevention:
  The prompts explicitly instruct the model to:
  1. Only answer from provided context (no parametric knowledge)
  2. Cite which document each fact comes from
  3. Say "I don't have sufficient data" if context doesn't support the answer
  4. Never invent property values or statistics

  The post-processing step in chatbot.py additionally verifies that any
  dollar amounts in the response actually appear in the retrieved documents.
"""

from enum import Enum


class QueryType(str, Enum):
    VALUATION = "valuation"
    COMPARISON = "comparison"
    MARKET_ANALYSIS = "market_analysis"
    GENERAL = "general"


SYSTEM_PROMPT = """You are a property valuation assistant for the Austin, Texas real estate market. \
You help users understand property values, market trends, and comparable sales data.

CRITICAL RULES:
1. ONLY use information from the provided context documents. Never use your general knowledge about real estate.
2. ALWAYS cite which document each piece of information comes from, using the format [Source: filename].
3. If the context does not contain enough information to answer the question, clearly state: \
"I don't have sufficient data in the available documents to answer this question."
4. Never invent or estimate property values unless the data explicitly supports it.
5. When discussing prices, always specify whether it's an assessed value, listing price, sale price, or estimated value.
6. Present numerical data accurately — do not round or approximate unless the source data is already rounded."""


VALUATION_TEMPLATE = """Based on the following property documents, answer the user's valuation question.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Instructions:
- Focus on the specific property mentioned in the question.
- IMPORTANT: Look for explicitly stated values in the documents first. Appraisal reports contain
  a "Valuation Conclusion" section with a stated "estimated market value" — report THAT number.
  Do NOT compute your own estimate from price-per-square-foot or comparable sales unless the
  document does not already state a concluded value.
- Clearly distinguish between: assessed value (county tax assessment), listing price (asking price),
  estimated market value (appraiser's conclusion), and sale price (what it actually sold for).
- If comparable sales data is available, reference it to support the valuation.
- Cite each fact with [Source: filename].
- If the property is not found in the documents, say so clearly."""


COMPARISON_TEMPLATE = """Based on the following property documents, answer the user's comparison question.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Instructions:
- Compare the specific properties or criteria mentioned in the question.
- Use a structured comparison: list key attributes side by side (price, size, bedrooms, etc.).
- Highlight significant differences and similarities.
- Reference comparable sales if relevant.
- Cite each fact with [Source: filename]."""


MARKET_ANALYSIS_TEMPLATE = """Based on the following market data documents, answer the user's market analysis question.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Instructions:
- Focus on market trends, statistics, and neighborhood-level data.
- Include specific numbers: average prices, days on market, inventory levels, year-over-year changes.
- Note which direction the market is trending (rising, stable, declining).
- If data covers multiple neighborhoods, compare them.
- Cite each fact with [Source: filename]."""


GENERAL_TEMPLATE = """Based on the following documents, answer the user's question about the Austin real estate market.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Instructions:
- Answer based only on the provided context.
- If the user asks about properties matching certain criteria, present whatever matching
  properties you find in the context — even if there is only one match, that is still
  useful information worth presenting in detail.
- Cite each fact with [Source: filename].
- If the question cannot be answered at all from the provided context, clearly state that."""


TEMPLATES = {
    QueryType.VALUATION: VALUATION_TEMPLATE,
    QueryType.COMPARISON: COMPARISON_TEMPLATE,
    QueryType.MARKET_ANALYSIS: MARKET_ANALYSIS_TEMPLATE,
    QueryType.GENERAL: GENERAL_TEMPLATE,
}


def classify_query(query: str) -> QueryType:
    """
    Classify a user query to select the appropriate prompt template.

    This ensures the LLM receives task-specific instructions that
    guide it toward the most relevant and well-structured answer.
    """
    q = query.lower()

    if any(w in q for w in ["value", "worth", "price of", "appraisal", "assessed", "estimate"]):
        return QueryType.VALUATION
    if any(w in q for w in ["compare", "comparison", "versus", "vs", "difference between", "similar"]):
        return QueryType.COMPARISON
    if any(w in q for w in ["market", "trend", "average", "median", "inventory", "going up", "going down"]):
        return QueryType.MARKET_ANALYSIS
    return QueryType.GENERAL


def format_context(search_results: list) -> str:
    """Format search results into context for the LLM prompt."""
    context_parts = []
    for i, result in enumerate(search_results, 1):
        source = result.document_source or "unknown"
        doc_type = result.document_type or "unknown"
        score = result.score

        context_parts.append(
            f"--- Document {i} [Source: {source}] (Type: {doc_type}, Relevance: {score:.2f}) ---\n"
            f"{result.text}\n"
        )
    return "\n".join(context_parts)


def build_prompt(query: str, search_results: list) -> tuple[str, str]:
    """
    Build the complete system + user prompt for the LLM.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    query_type = classify_query(query)
    template = TEMPLATES[query_type]
    context = format_context(search_results)

    user_prompt = template.format(context=context, question=query)
    return SYSTEM_PROMPT, user_prompt
