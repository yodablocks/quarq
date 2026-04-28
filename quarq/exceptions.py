"""quarq custom exception hierarchy."""


class QuarqError(Exception):
    """Base exception for all quarq errors."""


class ConfigError(QuarqError):
    """Raised when config cannot be read, written, or validated."""


class ProviderError(QuarqError):
    """Raised when a data provider fetch fails."""


class RAGError(QuarqError):
    """Raised when a RAG operation fails."""
