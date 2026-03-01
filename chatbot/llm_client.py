"""
LLM client abstraction with Gemini primary and OpenAI fallback.

PRODUCTION NOTE — Gemini via Vertex AI:
  In production, Gemini Pro via Vertex AI provides:
    - Enterprise-grade SLAs and data residency
    - Integration with Google Cloud IAM
    - Built-in content safety filters
    - Grounding with Google Search (optional)
    - Long context window (up to 2M tokens)
  For a property valuation chatbot, Gemini's grounding feature could
  cross-reference answers with live market data from Google Search.

LOCAL FALLBACK — OpenAI GPT-4o-mini:
  Used for the live demo to avoid cloud dependency.
  Provides reliable, fast responses for interview presentation.
"""

import logging
from typing import Generator, Protocol

import config

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]: ...


class OpenAIClient:
    """OpenAI GPT client for local demo."""

    def __init__(self, model: str = config.OPENAI_MODEL, api_key: str = config.OPENAI_API_KEY):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a complete response."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """Stream response tokens for real-time display."""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiClient:
    """
    Google Gemini via Vertex AI SDK.

    Requires:
      - GCP project with Vertex AI API enabled
      - Application Default Credentials configured

    IMPORTANT: Gemini's API handles system instructions differently from OpenAI.
    The system_instruction is set when creating the model, NOT mixed into the
    content list. We create a new model instance per call since the system prompt
    changes based on query type (valuation vs. market trends vs. comparison).
    """

    def __init__(
        self,
        model: str = config.GEMINI_MODEL,
        project: str = config.GCP_PROJECT_ID,
        location: str = config.GCP_LOCATION,
    ):
        import vertexai
        vertexai.init(project=project, location=location)
        self._model_name = model
        logger.info(f"Initialized Gemini client: {model} in {project}/{location}")

    def _get_model(self, system_prompt: str):
        """Create a GenerativeModel with the system instruction baked in."""
        from vertexai.generative_models import GenerativeModel
        return GenerativeModel(
            self._model_name,
            system_instruction=system_prompt,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using Gemini with proper system instruction."""
        from vertexai.generative_models import GenerationConfig

        model = self._get_model(system_prompt)
        response = model.generate_content(
            user_prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text

    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """Stream response from Gemini."""
        from vertexai.generative_models import GenerationConfig

        model = self._get_model(system_prompt)
        response = model.generate_content(
            user_prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text


def get_llm_client() -> LLMClient:
    """Factory function — returns the configured LLM client."""
    if config.USE_GEMINI:
        logger.info("Using Gemini Pro via Vertex AI (cloud mode)")
        return GeminiClient()
    else:
        logger.info("Using OpenAI GPT (demo mode)")
        return OpenAIClient()
