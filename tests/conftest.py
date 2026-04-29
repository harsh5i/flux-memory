from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flux.storage import FluxStore

# Make tests/mocks.py importable as `import mocks` from all test files.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def store(tmp_path: Path) -> FluxStore:
    db = tmp_path / "flux.db"
    s = FluxStore(db)
    try:
        yield s
    finally:
        s.close()
