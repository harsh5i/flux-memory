"""Tests for lazy embedding backend startup behavior."""
from __future__ import annotations

import builtins

from flux.embedding import SentenceTransformerBackend


def test_sentence_transformer_backend_does_not_import_model_on_init(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise AssertionError("sentence_transformers should load on first embed, not init")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    backend = SentenceTransformerBackend("lazy-test-model")

    assert backend.model_name == "lazy-test-model"
    assert backend._model is None
