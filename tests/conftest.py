from __future__ import annotations

from pathlib import Path

import pytest

from flux.storage import FluxStore


@pytest.fixture
def store(tmp_path: Path) -> FluxStore:
    db = tmp_path / "flux.db"
    s = FluxStore(db)
    try:
        yield s
    finally:
        s.close()
