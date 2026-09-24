"""Shared pytest configuration.

Adds demo/ to sys.path so the Open WebUI tool classes can be imported by
tests. demo/ is not part of the installed wheel (see [tool.hatch.build]),
so it is not importable as a package.

Also clears every environment variable quarq reads, so the suite behaves the
same on a developer machine with secrets exported as it does in CI.
"""

import sys
from pathlib import Path

import pytest

from quarq.config import _ENV_SECRETS

_DEMO = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

# Read outside config.py: ANTHROPIC_API_KEY selects the Claude fallback in
# quarq.llm, QUARQ_API_URL points the Open WebUI tools at a server.
_OTHER_ENV_VARS = ("ANTHROPIC_API_KEY", "QUARQ_API_URL")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset quarq's environment variables; tests that need one set it themselves.

    Args:
        monkeypatch: pytest fixture; restores the original environment afterwards.
    """
    for name in (*_ENV_SECRETS, *_OTHER_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
