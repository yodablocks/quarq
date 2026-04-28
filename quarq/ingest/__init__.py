"""quarq ingest layer — data provider registry."""

from __future__ import annotations

from quarq.exceptions import ProviderError
from quarq.ingest.base import BaseProvider
from quarq.ingest.ecb import ECBProvider
from quarq.ingest.equity import EquityProvider
from quarq.ingest.fred import FREDProvider
from quarq.ingest.oecd import OECDProvider

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "equity": EquityProvider,
    "fred": FREDProvider,
    "ecb": ECBProvider,
    "oecd": OECDProvider,
}


def get_provider(name: str) -> BaseProvider:
    """Instantiate and return a provider by name.

    Args:
        name: Provider name key (e.g. 'equity', 'fred', 'ecb', 'oecd').

    Returns:
        A new instance of the requested BaseProvider subclass.

    Raises:
        ProviderError: If name is not in PROVIDER_REGISTRY.
    """
    if name not in PROVIDER_REGISTRY:
        raise ProviderError(
            f"Unknown provider: '{name}'. Available: {list(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[name]()
