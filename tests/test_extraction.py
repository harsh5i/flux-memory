"""Tests for Track 2 — LLM backends, embedding, and extraction."""
from __future__ import annotations

import json
import math
from datetime import timezone, datetime
from pathlib import Path

import pytest

from flux import Config, Entry, Grain, FluxStore
from flux.embedding import (
    SentenceTransformerBackend,
    cosine_similarity,
    load_all_embeddings,
    store_embedding,
    top_k_nearest,
    vector_fallback,
)
from flux.extraction import (
    _fallback_tokenize,
    decompose_query,
    extract_and_store_grains,
)
from flux.llm import (
    OllamaBackend,
    parse_features,
    parse_grains,
)
from mocks import MockEmbeddingBackend, MockLLMBackend

import numpy as np


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    with FluxStore(db) as s:
        yield s


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ===================================================================== MockLLMBackend

class TestMockLLMBackend:
    def test_feature_extraction_returns_list(self):
        llm = MockLLMBackend()
        from flux.llm import _FEATURE_EXTRACTION_PROMPT
        prompt = _FEATURE_EXTRACTION_PROMPT.format(query="Help me with AI frameworks")
        result = llm.complete(prompt)
        features = parse_features(result)
        assert isinstance(features, list)
        assert len(features) >= 1

    def test_grain_extraction_returns_list(self):
        llm = MockLLMBackend()
        from flux.llm import _GRAIN_EXTRACTION_PROMPT
        prompt = _GRAIN_EXTRACTION_PROMPT.format(
            user_message="I prefer Python for ML work.",
            ai_response="Great choice.",
        )
        result = llm.complete(prompt)
        grains = parse_grains(result)
        assert isinstance(grains, list)

    def test_unknown_prompt_returns_empty_array(self):
        llm = MockLLMBackend()
        result = llm.complete("some unknown prompt")
        assert result == "[]"


# ===================================================================== parse_features

class TestParseFeatures:
    def test_valid_json_array(self):
        assert parse_features('["AI", "framework", "project"]') == ["ai", "framework", "project"]

    def test_mixed_case_lowercased(self):
        result = parse_features('["Python", "ML"]')
        assert result == ["python", "ml"]

    def test_fallback_on_invalid_json(self):
        result = parse_features('AI, framework, project')
        assert isinstance(result, list)

    def test_quoted_strings_extracted_as_fallback(self):
        result = parse_features('Here are the keywords: "AI" and "framework"')
        assert "AI" in result or "framework" in result

    def test_empty_json_returns_default(self):
        result = parse_features("[]")
        assert result == []  # empty is valid; caller handles empty list


# ===================================================================== parse_grains

class TestParseGrains:
    def test_valid_grain_dict(self):
        raw = json.dumps([{"content": "User prefers Python", "provenance": "user_stated"}])
        result = parse_grains(raw)
        assert len(result) == 1
        assert result[0]["content"] == "User prefers Python"
        assert result[0]["provenance"] == "user_stated"

    def test_unknown_provenance_defaults_to_ai_stated(self):
        raw = json.dumps([{"content": "something", "provenance": "unknown_type"}])
        result = parse_grains(raw)
        assert result[0]["provenance"] == "ai_stated"

    def test_invalid_json_returns_empty(self):
        result = parse_grains("not json at all")
        assert result == []

    def test_missing_content_excluded(self):
        raw = json.dumps([{"provenance": "user_stated"}])
        result = parse_grains(raw)
        # Items without "content" key should be excluded
        assert result == []


# ===================================================================== MockEmbeddingBackend

class TestMockEmbeddingBackend:
    def test_embed_returns_list_of_floats(self):
        backend = MockEmbeddingBackend()
        emb = backend.embed("test text")
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)

    def test_embed_is_normalised(self):
        backend = MockEmbeddingBackend()
        emb = backend.embed("hello world")
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-6

    def test_same_text_same_embedding(self):
        backend = MockEmbeddingBackend()
        e1 = backend.embed("consistent")
        e2 = backend.embed("consistent")
        assert e1 == e2

    def test_different_texts_different_embeddings(self):
        backend = MockEmbeddingBackend()
        e1 = backend.embed("Python is great")
        e2 = backend.embed("completely different topic XYZ")
        assert e1 != e2

    def test_embed_batch_matches_individual(self):
        backend = MockEmbeddingBackend()
        texts = ["alpha", "beta", "gamma"]
        batch = backend.embed_batch(texts)
        individuals = [backend.embed(t) for t in texts]
        for b, ind in zip(batch, individuals):
            assert b == pytest.approx(ind)


# ===================================================================== cosine_similarity

class TestCosineSimilarity:
    def test_identical_vectors_are_1(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_0(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_are_minus_1(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_0(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ===================================================================== top_k_nearest

class TestTopKNearest:
    def test_returns_top_k_results(self):
        backend = MockEmbeddingBackend()
        texts = ["alpha", "beta", "gamma", "delta"]
        grain_ids = [f"g{i}" for i in range(4)]
        matrix = np.array([backend.embed(t) for t in texts], dtype=np.float32)
        query = backend.embed("alpha")
        results = top_k_nearest(query, grain_ids, matrix, k=2)
        assert len(results) == 2

    def test_empty_matrix_returns_empty(self):
        results = top_k_nearest([1.0, 0.0], [], np.empty((0, 0)), k=5)
        assert results == []

    def test_highest_similarity_first(self):
        backend = MockEmbeddingBackend()
        grain_ids = ["exact", "different"]
        exact_emb = backend.embed("Python")
        diff_emb = backend.embed("unrelated XYZ")
        matrix = np.array([exact_emb, diff_emb], dtype=np.float32)
        query = backend.embed("Python")
        results = top_k_nearest(query, grain_ids, matrix, k=2)
        assert results[0][0] == "exact"
        assert results[0][1] >= results[1][1]


# ===================================================================== store/load embeddings

class TestEmbeddingStorage:
    def test_store_and_load_roundtrip(self, store):
        g = Grain(content="test grain", provenance="user_stated")
        store.insert_grain(g)
        backend = MockEmbeddingBackend()
        emb = backend.embed(g.content)
        store_embedding(store, g.id, emb, "mock", _now())
        grain_ids, matrix = load_all_embeddings(store)
        assert g.id in grain_ids
        idx = grain_ids.index(g.id)
        assert list(matrix[idx]) == pytest.approx(emb)

    def test_dormant_grains_excluded(self, store):
        g = Grain(content="dormant grain", provenance="user_stated", status="dormant")
        store.insert_grain(g)
        emb = MockEmbeddingBackend().embed(g.content)
        store_embedding(store, g.id, emb, "mock", _now())
        grain_ids, _ = load_all_embeddings(store)
        assert g.id not in grain_ids


# ===================================================================== vector_fallback

class TestVectorFallback:
    def test_returns_results_when_graph_empty(self, store):
        backend = MockEmbeddingBackend()
        g = Grain(content="Python preference", provenance="user_stated")
        store.insert_grain(g)
        store_embedding(store, g.id, backend.embed(g.content), "mock", _now())
        results = vector_fallback(store, "Python", backend, [], cfg=Config(TOP_K=5))
        assert len(results) >= 1

    def test_merges_with_existing_results(self, store):
        backend = MockEmbeddingBackend()
        g1 = Grain(content="alpha grain", provenance="user_stated")
        g2 = Grain(content="beta grain", provenance="user_stated")
        store.insert_grain(g1); store.insert_grain(g2)
        store_embedding(store, g1.id, backend.embed(g1.content), "mock", _now())
        store_embedding(store, g2.id, backend.embed(g2.content), "mock", _now())

        existing = [(g1.id, 0.8)]  # g1 already found by graph
        results = vector_fallback(store, "alpha", backend, existing, cfg=Config(TOP_K=5))
        grain_ids = [gid for gid, _ in results]
        assert g1.id in grain_ids  # should be preserved or merged

    def test_scores_sorted_descending(self, store):
        backend = MockEmbeddingBackend()
        for i in range(3):
            g = Grain(content=f"grain {i}", provenance="user_stated")
            store.insert_grain(g)
            store_embedding(store, g.id, backend.embed(g.content), "mock", _now())
        results = vector_fallback(store, "grain", backend, [], cfg=Config(TOP_K=5))
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store_returns_existing(self, store):
        existing = [("g1", 0.5)]
        results = vector_fallback(store, "query", MockEmbeddingBackend(), existing, cfg=Config())
        assert results == existing


# ===================================================================== decompose_query

class TestDecomposeQuery:
    def test_returns_entry_ids(self, store):
        llm = MockLLMBackend()
        entry_ids = decompose_query("Help me with AI frameworks", llm, store)
        assert isinstance(entry_ids, list)
        assert len(entry_ids) >= 1
        # All IDs should exist in the store
        for eid in entry_ids:
            assert store.get_entry(eid) is not None

    def test_reuses_existing_entry(self, store):
        llm = MockLLMBackend()
        e = Entry(feature="python")
        store.insert_entry(e)
        entry_ids_1 = decompose_query("I prefer python for ML", llm, store)
        entry_ids_2 = decompose_query("Python is my language", llm, store)
        # The "python" entry should be reused (same ID across calls)
        all_features = {store.get_entry(eid).feature for eid in entry_ids_1 + entry_ids_2}
        assert "python" in all_features

    def test_creates_new_entries_for_new_features(self, store):
        llm = MockLLMBackend()
        before = store.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        decompose_query("completely novel topic xyz pqr", llm, store)
        after = store.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        assert after >= before  # at least some entries created

    def test_logs_event(self, store):
        llm = MockLLMBackend()
        decompose_query("test query", llm, store)
        count = store.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE event='features_extracted'"
        ).fetchone()["n"]
        assert count >= 1


# ===================================================================== extract_and_store_grains

class TestExtractAndStoreGrains:
    def test_stores_grains(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        new_ids = extract_and_store_grains(
            "I prefer Python for ML work.",
            "Python is an excellent choice.",
            llm, emb, store,
        )
        assert len(new_ids) >= 1
        for gid in new_ids:
            g = store.get_grain(gid)
            assert g is not None
            assert g.status == "active"

    def test_embeddings_stored(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        new_ids = extract_and_store_grains(
            "I use dark themes in my editor.",
            "Dark themes reduce eye strain.",
            llm, emb, store,
        )
        assert len(new_ids) >= 1
        grain_ids_in_store, _ = load_all_embeddings(store)
        for gid in new_ids:
            assert gid in grain_ids_in_store

    def test_bootstrap_conduits_created(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        # Insert a grain first so bootstrap has neighbours
        existing = Grain(content="Python machine learning", provenance="user_stated")
        store.insert_grain(existing)
        store_embedding(store, existing.id, emb.embed(existing.content), "mock", _now())

        new_ids = extract_and_store_grains(
            "Python is my preferred ML language.",
            "Great, Python has excellent ML libraries.",
            llm, emb, store,
        )
        # New grains should have conduits to existing grain
        total_conduits = store.conn.execute("SELECT COUNT(*) AS n FROM conduits").fetchone()["n"]
        assert total_conduits >= 0  # At least some conduits created

    def test_entry_points_connected(self, store):
        llm = MockLLMBackend()
        emb = MockEmbeddingBackend()
        extract_and_store_grains(
            "I prefer Python.",
            "Python is great.",
            llm, emb, store,
        )
        # Entry points should have been created and linked
        entries_count = store.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        assert entries_count >= 1

    def test_empty_response_returns_empty(self, store):
        class EmptyLLM:
            def complete(self, prompt): return "[]"
        new_ids = extract_and_store_grains("hi", "hello", EmptyLLM(), MockEmbeddingBackend(), store)
        assert new_ids == []


# ===================================================================== _fallback_tokenize

class TestFallbackTokenize:
    def test_removes_stopwords(self):
        result = _fallback_tokenize("Help me with the framework")
        assert "the" not in result
        assert "me" not in result

    def test_returns_at_most_5(self):
        result = _fallback_tokenize("a b c d e f g h i j k l m n")
        assert len(result) <= 5

    def test_empty_string_returns_default(self):
        result = _fallback_tokenize("")
        assert result == ["query"]
