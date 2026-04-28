"""Tests for quarq ingest layer."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest


def test_base_provider_cannot_be_instantiated() -> None:
    """BaseProvider is abstract and raises TypeError on direct instantiation."""
    from quarq.ingest.base import BaseProvider

    with pytest.raises(TypeError):
        BaseProvider()  # type: ignore[abstract]
