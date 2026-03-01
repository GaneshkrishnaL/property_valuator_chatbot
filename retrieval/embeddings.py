"""
Embedding model abstraction with Vertex AI primary and local fallback.

PRODUCTION NOTE — Vertex AI Embeddings:
  text-embedding-004 is Google's latest embedding model with 768 dimensions.
  It supports task_type hints (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY) that
  optimize embeddings for search scenarios. In production with Vertex AI,
  you also get automatic batching and rate limit handling.

LOCAL FALLBACK:
  sentence-transformers/all-MiniLM-L6-v2 provides 384-dim embeddings that
  run entirely on CPU with no API key required. Ideal for live demos where
  you don't want cloud dependency risk.
"""

import logging
from typing import Protocol

import config

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding models — enables clean swapping."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...


class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers (no API key needed)."""

    def __init__(self, model_name: str = config.LOCAL_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded {model_name} ({self._dimension} dimensions)")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


class VertexAIEmbeddingModel:
    """
    Google Vertex AI text-embedding-004 model.

    Requires:
      - GCP project with Vertex AI API enabled
      - Application Default Credentials configured
      - pip install google-cloud-aiplatform
    """

    def __init__(
        self,
        model_name: str = config.VERTEX_EMBEDDING_MODEL,
        project: str = config.GCP_PROJECT_ID,
        location: str = config.GCP_LOCATION,
    ):
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)

        from vertexai.language_models import TextEmbeddingModel as VertexEmbModel
        logger.info(f"Initializing Vertex AI embedding model: {model_name}")
        self._model = VertexEmbModel.from_pretrained(model_name)
        self._dimension = 768  # text-embedding-004 output dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed documents with RETRIEVAL_DOCUMENT task type for optimal search.

        Vertex AI has a batch limit of ~250 texts per API call.
        We batch automatically so callers don't have to worry about it.
        """
        from vertexai.language_models import TextEmbeddingInput

        BATCH_SIZE = 200  # stay safely under the 250 limit
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in batch]
            embeddings = self._model.get_embeddings(inputs)
            all_embeddings.extend([e.values for e in embeddings])
            if len(texts) > BATCH_SIZE:
                logger.info(f"Embedded batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}")

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with RETRIEVAL_QUERY task type."""
        from vertexai.language_models import TextEmbeddingInput
        inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")]
        embeddings = self._model.get_embeddings(inputs)
        return embeddings[0].values


def get_embedding_model() -> EmbeddingModel:
    """
    Factory function — returns the configured embedding model.

    Set USE_VERTEX_AI=true in environment to use Google Cloud embeddings.
    Defaults to local sentence-transformers for demo reliability.
    """
    if config.USE_VERTEX_AI:
        logger.info("Using Vertex AI embeddings (cloud mode)")
        return VertexAIEmbeddingModel()
    else:
        logger.info("Using local sentence-transformers embeddings (demo mode)")
        return LocalEmbeddingModel()
