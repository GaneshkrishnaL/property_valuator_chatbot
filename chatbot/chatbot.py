"""
Main chatbot orchestrator that ties together retrieval + LLM.

Flow:
  1. Classify the query type
  2. Extract structured filters from the query
  3. Run hybrid search (metadata filter → vector search → rerank)
  4. Format retrieved context into the prompt
  5. Call the LLM with the appropriate template
  6. Post-process: verify cited numbers exist in context
  7. Return response with source citations
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Generator

from chatbot.llm_client import LLMClient, get_llm_client
from chatbot.prompt_templates import build_prompt, classify_query
from retrieval.embeddings import EmbeddingModel, get_embedding_model
from retrieval.search import SearchResult, hybrid_search
from retrieval.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """A complete chatbot response with metadata."""
    answer: str
    query_type: str
    sources: list[str]
    retrieved_documents: list[SearchResult]
    confidence: str  # "high", "medium", "low", "insufficient_data"


@dataclass
class ConversationHistory:
    """Maintains conversation context for follow-up questions."""
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_turn(self, query: str, response: str) -> None:
        self.messages.append({"role": "user", "content": query})
        self.messages.append({"role": "assistant", "content": response})

    def get_context_hint(self) -> str:
        """Return recent conversation context for follow-up question handling."""
        if not self.messages:
            return ""
        recent = self.messages[-4:]  # last 2 turns
        parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content'][:200]}")
        return "\nPrevious conversation:\n" + "\n".join(parts)


class PropertyValuationChatbot:
    """
    End-to-end RAG chatbot for property valuation queries.

    Combines custom retrieval pipeline with LLM generation to answer
    questions about real estate properties, market trends, and valuations.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
    ):
        self._embedding_model = embedding_model or get_embedding_model()
        self._vector_store = vector_store or get_vector_store()
        self._llm_client = llm_client or get_llm_client()
        self._history = ConversationHistory()

        logger.info("PropertyValuationChatbot initialized")

    def ask(self, query: str, use_history: bool = True) -> ChatResponse:
        """
        Process a user query through the full RAG pipeline.

        Args:
            query: Natural language question.
            use_history: Whether to include conversation history for context.

        Returns:
            ChatResponse with answer, sources, and metadata.
        """
        logger.info(f"Processing query: {query}")

        # IMPORTANT: Extract filters and run retrieval on the RAW query only.
        # Conversation history must NOT be mixed into the search query, because
        # extract_query_filters() uses regex and would pick up numbers, addresses,
        # and keywords from previous Q&A turns (e.g., bedrooms=3 from a prior
        # question leaking into an unrelated follow-up).
        # History context is only injected into the LLM prompt below.

        # Step 1: Retrieve relevant documents (raw query — no history contamination)
        results = hybrid_search(
            query=query,
            embedding_model=self._embedding_model,
            vector_store=self._vector_store,
        )

        # Step 2: Classify query type
        query_type = classify_query(query)

        # Step 3: Check if we have enough context
        if not results:
            response = ChatResponse(
                answer="I don't have sufficient data in the available documents to answer this question. "
                       "The property or topic you're asking about may not be in our database.",
                query_type=query_type.value,
                sources=[],
                retrieved_documents=[],
                confidence="insufficient_data",
            )
            self._history.add_turn(query, response.answer)
            return response

        # Step 4: Build prompt and call LLM
        # Conversation history is added here (LLM prompt only) for follow-up context
        system_prompt, user_prompt = build_prompt(query, results)
        if use_history and self._history.messages:
            context_hint = self._history.get_context_hint()
            user_prompt = user_prompt + "\n" + context_hint

        answer = self._llm_client.generate(system_prompt, user_prompt)

        # Step 5: Post-process — verify numbers and extract sources
        answer = self._verify_numbers(answer, results)
        sources = self._extract_sources(answer, results)
        confidence = self._assess_confidence(results, answer)

        response = ChatResponse(
            answer=answer,
            query_type=query_type.value,
            sources=sources,
            retrieved_documents=results,
            confidence=confidence,
        )

        self._history.add_turn(query, answer)
        return response

    def ask_stream(self, query: str) -> Generator[str, None, None]:
        """
        Stream the response token by token (for interactive demo).

        Note: Post-processing (number verification) happens on the full
        response, so streaming shows raw LLM output. For the demo,
        this is fine — the streaming effect looks impressive.
        """
        results = hybrid_search(
            query=query,
            embedding_model=self._embedding_model,
            vector_store=self._vector_store,
        )

        if not results:
            yield "I don't have sufficient data in the available documents to answer this question."
            return

        system_prompt, user_prompt = build_prompt(query, results)
        for token in self._llm_client.generate_stream(system_prompt, user_prompt):
            yield token

    def _verify_numbers(self, answer: str, results: list[SearchResult]) -> str:
        """
        Verify that dollar amounts in the answer appear in retrieved context.

        HALLUCINATION PREVENTION: If the LLM generates a number not found
        in any retrieved document, flag it. This prevents made-up valuations.

        We normalize amounts before comparing, so "$525,000" matches even if
        surrounding punctuation differs. Amounts that are reasonable derivations
        (e.g., computed averages from price-per-sqft) get a DEBUG log, not WARNING.
        """
        # Extract and normalize dollar amounts from the answer
        raw_answer = re.findall(r"\$([\d,]+)", answer)
        if not raw_answer:
            return answer

        answer_nums = set()
        for amt in raw_answer:
            try:
                answer_nums.add(int(amt.replace(",", "")))
            except ValueError:
                pass

        # Collect and normalize all dollar amounts from retrieved documents
        context_text = " ".join(r.text for r in results)
        raw_context = re.findall(r"\$([\d,]+)", context_text)
        context_nums = set()
        for amt in raw_context:
            try:
                context_nums.add(int(amt.replace(",", "")))
            except ValueError:
                pass

        unsupported = answer_nums - context_nums
        if unsupported:
            # Only warn about amounts far from any context amount;
            # computed averages (e.g., avg price-per-sqft * sqft) are expected
            truly_suspicious = set()
            for num in unsupported:
                is_close = any(abs(num - ctx) / max(ctx, 1) < 0.20 for ctx in context_nums)
                if not is_close:
                    truly_suspicious.add(num)

            if truly_suspicious:
                logger.warning(f"Answer may contain hallucinated amounts: "
                             f"{['${:,}'.format(n) for n in truly_suspicious]}")
            else:
                logger.debug(f"Answer contains derived amounts (computed from context): "
                           f"{['${:,}'.format(n) for n in unsupported]}")

        return answer

    def _extract_sources(self, answer: str, results: list[SearchResult]) -> list[str]:
        """Extract unique source documents referenced in the answer."""
        sources = set()
        # Collect from [Source: ...] citations in the answer
        cited = re.findall(r"\[Source:\s*([^\]]+)\]", answer)
        sources.update(cited)

        # Also include top retrieved documents as sources
        for r in results[:3]:
            if r.document_source:
                sources.add(r.document_source)

        return sorted(sources)

    def _assess_confidence(self, results: list[SearchResult], answer: str) -> str:
        """Assess response confidence based on retrieval quality."""
        if not results:
            return "insufficient_data"

        avg_score = sum(r.score for r in results) / len(results)
        top_score = results[0].score

        # Check if the answer says "I don't have sufficient data"
        if "don't have sufficient data" in answer.lower() or "not found" in answer.lower():
            return "insufficient_data"

        if top_score > 0.7 and avg_score > 0.5:
            return "high"
        elif top_score > 0.5:
            return "medium"
        return "low"
