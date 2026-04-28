"""Sentence-transformers embedding backend with multilingual-e5 prefix enforcement."""

from __future__ import annotations

import logging

from quarq.exceptions import RAGError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
_BATCH_SIZE = 32


class Embedder:
    """Local embedding model wrapper for multilingual-e5-large.

    Loads the model lazily on the first call to embed() or embed_query().
    Enforces the required passage/query prefix scheme for multilingual-e5 models.

    Args:
        model_name: HuggingFace model identifier. Defaults to multilingual-e5-large.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._model = None  # lazy-loaded

    def _load_model(self) -> None:
        """Load the sentence-transformers model on first use.

        Raises:
            RAGError: If the model cannot be loaded.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            raise RAGError(f"Failed to load embedding model {self._model_name!r}: {exc}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of passage texts.

        Prepends 'passage: ' to each text as required by multilingual-e5 models.

        Args:
            texts: List of raw text strings to embed.

        Returns:
            List of embedding vectors (one per input text).

        Raises:
            RAGError: If model loading or encoding fails.
        """
        self._load_model()
        prefixed = [f"passage: {t}" for t in texts]
        try:
            vectors = self._model.encode(prefixed, batch_size=_BATCH_SIZE)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception as exc:
            raise RAGError(f"Embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Prepends 'query: ' as required by multilingual-e5 models (different from passages).

        Args:
            text: Raw query string.

        Returns:
            Single embedding vector as a list of floats.

        Raises:
            RAGError: If model loading or encoding fails.
        """
        self._load_model()
        prefixed = f"query: {text}"
        try:
            vectors = self._model.encode([prefixed], batch_size=1)
            v = vectors[0]
            return v.tolist() if hasattr(v, "tolist") else list(v)
        except Exception as exc:
            raise RAGError(f"Query embedding failed: {exc}") from exc
