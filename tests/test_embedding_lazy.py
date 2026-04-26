"""Tests for lazy embedding backend startup behavior."""
from __future__ import annotations

import builtins

from flux import Config, FluxStore
from flux.embedding import SentenceTransformerBackend
from flux.embedding import vector_fallback


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


def test_vector_fallback_does_not_embed_when_store_has_no_embeddings(tmp_path):
    class FailingEmbeddingBackend:
        def embed(self, text: str) -> list[float]:
            raise AssertionError("empty vector fallback should not load/embed")

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("empty vector fallback should not batch embed")

    with FluxStore(tmp_path / "test.db") as store:
        existing_results = [("existing-grain", 0.25)]

        results = vector_fallback(
            store,
            "query that would otherwise trigger cold model load",
            FailingEmbeddingBackend(),
            existing_results,
            cfg=Config(),
        )

    assert results == existing_results
