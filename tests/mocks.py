"""Test-only mock backends. Must never be imported from src/flux/.

These implementations provide deterministic, dependency-free substitutes for
the real LLM and embedding backends, enabling fast offline unit tests.
"""
from __future__ import annotations

import json
import math
import re


class MockLLMBackend:
    """Deterministic mock LLM backend for tests.

    Keyword extraction: returns the first 3 non-stopword tokens from the query.
    Grain extraction: returns one grain per sentence in the user message.
    """

    _STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "for", "of",
                  "to", "in", "on", "at", "by", "with", "and", "or", "but",
                  "how", "what", "when", "where", "why", "who", "which", "help",
                  "me", "my", "i", "you", "we", "they", "it", "that", "this"}

    def complete(self, prompt: str) -> str:
        if "Extract 2-5 key concept words" in prompt:
            return self._mock_features(prompt)
        if "Extract atomic facts" in prompt:
            return self._mock_grains(prompt)
        return "[]"

    def _mock_features(self, prompt: str) -> str:
        matches = re.findall(r'Query: "(.+?)"', prompt)
        query = matches[-1] if matches else prompt
        words = re.findall(r"[a-zA-Z]+", query.lower())
        keywords = [w for w in words if w not in self._STOPWORDS][:3]
        return json.dumps(keywords or ["query"])

    def _mock_grains(self, prompt: str) -> str:
        m = re.search(r"User message: (.+?)(?:\nAI response:)", prompt, re.DOTALL)
        user_text = m.group(1).strip() if m else ""
        sentences = [s.strip() for s in re.split(r"[.!?]", user_text) if s.strip()]
        grains = [
            {"content": s, "provenance": "user_stated"}
            for s in sentences[:3]
        ]
        return json.dumps(grains)


class MockEmbeddingBackend:
    """Deterministic pseudo-embeddings for tests. Hash-based, 16-dim, L2-normalised."""

    DIM = 16
    model_name = "mock-embedding"

    def embed(self, text: str) -> list[float]:
        rng = [((hash(text + str(i)) % 1000) / 1000.0) for i in range(self.DIM)]
        norm = math.sqrt(sum(x * x for x in rng)) or 1.0
        return [x / norm for x in rng]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
