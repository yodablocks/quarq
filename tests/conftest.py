"""Shared pytest configuration.

Adds demo/ to sys.path so the Open WebUI tool classes can be imported by
tests. demo/ is not part of the installed wheel (see [tool.hatch.build]),
so it is not importable as a package.
"""

import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent.parent / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))
