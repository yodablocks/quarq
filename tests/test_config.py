"""Tests for quarq.config module."""

from pathlib import Path

import pytest

from quarq.config import QuarqConfig, get_config_path, load_config, save_config


def test_load_config_creates_default_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config() creates ~/.quarq/config.toml with defaults when none exists."""
    config_file = tmp_path / ".quarq" / "config.toml"
    monkeypatch.setattr("quarq.config.get_config_path", lambda: config_file)

    assert not config_file.exists()
    cfg = load_config()

    assert config_file.exists()
    assert cfg.lmstudio.url == "http://192.168.0.193:1234/v1"
    assert cfg.lmstudio.auto_discover_model is True
    assert cfg.llm.reporting_backend == "lmstudio"
    assert cfg.llm.reporting_model == "qwen/qwen3.5-9b"
    assert cfg.llm.research_backend == "lmstudio"
    assert cfg.llm.research_model == "qwen/qwen3.6-27b"
    assert cfg.llm.fallback_backend == "claude"
    assert cfg.llm.fallback_model == "claude-sonnet-4-20250514"
    assert cfg.embedder.backend == "local"
    assert cfg.embedder.model == "intfloat/multilingual-e5-large"
    assert cfg.data.fred_api_key == ""
    assert cfg.data.fred_enabled is True
    assert cfg.data.ecb_enabled is True
    assert cfg.data.oecd_enabled is True
    assert cfg.rag.chunk_size == 512
    assert cfg.rag.chunk_overlap == 64
    assert cfg.rag.top_k == 5
    assert cfg.rag.min_similarity == 0.35
    assert cfg.portfolio.default_benchmark == "^FCHI"
    assert cfg.portfolio.default_currency == "EUR"
    assert cfg.portfolio.risk_free_rate_source == "fred"
    assert cfg.portfolio.risk_free_rate_fallback == 0.03


def test_save_config_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_config() writes and load_config() reads back the same values."""
    config_file = tmp_path / ".quarq" / "config.toml"
    monkeypatch.setattr("quarq.config.get_config_path", lambda: config_file)

    cfg = load_config()
    cfg.data.fred_api_key = "TEST_KEY_123"
    cfg.rag.top_k = 10
    cfg.portfolio.risk_free_rate_fallback = 0.05
    save_config(cfg)

    cfg2 = load_config()
    assert cfg2.data.fred_api_key == "TEST_KEY_123"
    assert cfg2.rag.top_k == 10
    assert cfg2.portfolio.risk_free_rate_fallback == 0.05


def test_get_config_path_returns_path() -> None:
    """get_config_path() returns a Path object pointing at ~/.quarq/config.toml."""
    path = get_config_path()
    assert isinstance(path, Path)
    assert path.name == "config.toml"
    assert path.parent.name == ".quarq"


def test_env_var_overrides_fred_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FRED_API_KEY in the environment wins over the value stored on disk."""
    config_file = tmp_path / ".quarq" / "config.toml"
    monkeypatch.setattr("quarq.config.get_config_path", lambda: config_file)

    cfg = QuarqConfig()
    cfg.data.fred_api_key = "ON_DISK_KEY"
    save_config(cfg)

    monkeypatch.setenv("FRED_API_KEY", "FROM_ENV_KEY")
    assert load_config().data.fred_api_key == "FROM_ENV_KEY"


def test_env_var_secret_is_not_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A secret supplied via the environment is never written to config.toml."""
    config_file = tmp_path / ".quarq" / "config.toml"
    monkeypatch.setattr("quarq.config.get_config_path", lambda: config_file)
    monkeypatch.setenv("FRED_API_KEY", "SUPER_SECRET")

    cfg = load_config()

    assert cfg.data.fred_api_key == "SUPER_SECRET"
    assert "SUPER_SECRET" not in config_file.read_text()


def test_no_env_var_leaves_disk_value_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without FRED_API_KEY set, the on-disk value is used unchanged."""
    config_file = tmp_path / ".quarq" / "config.toml"
    monkeypatch.setattr("quarq.config.get_config_path", lambda: config_file)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    cfg = QuarqConfig()
    cfg.data.fred_api_key = "ON_DISK_KEY"
    save_config(cfg)

    assert load_config().data.fred_api_key == "ON_DISK_KEY"


def test_empty_env_var_does_not_clobber_disk_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty FRED_API_KEY is ignored rather than blanking the stored key."""
    config_file = tmp_path / ".quarq" / "config.toml"
    monkeypatch.setattr("quarq.config.get_config_path", lambda: config_file)
    monkeypatch.setenv("FRED_API_KEY", "")

    cfg = QuarqConfig()
    cfg.data.fred_api_key = "ON_DISK_KEY"
    save_config(cfg)

    assert load_config().data.fred_api_key == "ON_DISK_KEY"
