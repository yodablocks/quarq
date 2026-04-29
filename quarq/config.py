"""quarq configuration management.

Loads and saves ~/.quarq/config.toml using Pydantic v2 models.
Creates the file with defaults on first run if missing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

import tomli_w
from pydantic import BaseModel, Field

from quarq.exceptions import ConfigError


class LMStudioConfig(BaseModel):
    """LM Studio local inference server settings."""

    url: str = "http://192.168.0.193:1234/v1"
    auto_discover_model: bool = True


class LLMConfig(BaseModel):
    """LLM agent routing configuration."""

    reporting_backend: str = "lmstudio"
    reporting_model: str = "qwen/qwen3.5-9b"
    research_backend: str = "lmstudio"
    research_model: str = "qwen/qwen3.6-27b"
    fallback_backend: str = "claude"
    fallback_model: str = "claude-sonnet-4-20250514"


class EmbedderConfig(BaseModel):
    """Embedding model configuration."""

    backend: str = "local"
    model: str = "intfloat/multilingual-e5-large"


class DataConfig(BaseModel):
    """Data provider configuration."""

    fred_api_key: str = ""
    fred_enabled: bool = True
    ecb_enabled: bool = True
    oecd_enabled: bool = True


class RAGConfig(BaseModel):
    """RAG retrieval configuration."""

    chroma_path: str = "~/.quarq/chroma"
    corpus_paths: list[str] = Field(default_factory=list)
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    min_similarity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35


class PortfolioConfig(BaseModel):
    """Portfolio analytics configuration."""

    default_benchmark: str = "^FCHI"
    default_currency: str = "EUR"
    risk_free_rate_source: str = "fred"
    risk_free_rate_fallback: float = 0.03


class APIConfig(BaseModel):
    """FastAPI server settings."""

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    admin_key: str = ""


class QuarqConfig(BaseModel):
    """Top-level quarq configuration."""

    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def get_config_path() -> Path:
    """Return the path to ~/.quarq/config.toml."""
    return Path.home() / ".quarq" / "config.toml"


def load_config() -> QuarqConfig:
    """Load config from disk, creating with defaults if not found.

    Args:
        None

    Returns:
        Parsed QuarqConfig with all sections populated.

    Raises:
        ConfigError: If the file exists but cannot be parsed.
    """
    path = get_config_path()
    if not path.exists():
        cfg = QuarqConfig()
        save_config(cfg)
        return cfg
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return QuarqConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc


def save_config(config: QuarqConfig) -> None:
    """Write QuarqConfig back to ~/.quarq/config.toml.

    Args:
        config: The QuarqConfig instance to persist.

    Returns:
        None

    Raises:
        ConfigError: If the file cannot be written.
    """
    path = get_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump()
        with path.open("wb") as f:
            tomli_w.dump(data, f)
    except Exception as exc:
        raise ConfigError(f"Failed to save config to {path}: {exc}") from exc
