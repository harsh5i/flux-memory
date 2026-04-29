"""LLM backend abstraction for query decomposition and grain extraction (Track 2).

The spec requires a local 7B-8B instruction-tuned LLM for two purposes:
  1. Query decomposition (feature extractor): extracts 2-5 keywords per query.
  2. Grain extraction: reads (user_message + ai_response) and emits atomic facts.

The implementation is backend-agnostic: callers use the ``LLMBackend`` protocol.
``OllamaBackend`` is the default production backend (Ollama REST API).
Test backends live in tests/mocks.py and must not be imported from production code.

Backend selection via Config:
  cfg.LLM_BASE_URL  — e.g. "http://localhost:11434" for Ollama
  cfg.LLM_MODEL     — e.g. "llama3.1:8b"
  cfg.LLM_TIMEOUT_SECONDS

To use a different inference server (llama.cpp, LM Studio, etc.) that exposes
an OpenAI-compatible /v1/completions endpoint, set LLM_BASE_URL to that server
and subclass or swap the backend implementation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Prompts are kept here (not in extraction.py) so the LLM layer owns them.
# Both are short enough to fit in a 512-token context window.

_FEATURE_EXTRACTION_PROMPT = """\
Extract 2-5 key concept words from the following query. Return ONLY a JSON array of lowercase strings, nothing else.

Examples:
Query: "Help me pick a framework for an AI project"
Response: ["framework", "AI", "project"]

Query: "What was the deadline we discussed for VMO2?"
Response: ["deadline", "VMO2"]

Query: "{query}"
Response:"""

_GRAIN_EXTRACTION_PROMPT = """\
Extract atomic facts from the following conversation turn. Each fact should be a single, self-contained statement in plain English. Return ONLY a JSON array of objects with these fields:
  "content": the fact as a string
  "provenance": one of "user_stated", "ai_stated", "ai_inferred", "external_source"

Rules:
- One fact per item. No compound facts.
- Only extract facts that are worth remembering long-term.
- "user_stated": the user directly stated this.
- "ai_stated": the AI directly asserted this as fact.
- "ai_inferred": the AI reasoned or concluded this.
- "external_source": a cited document or source.
- Return [] if there are no notable facts to extract.

User message: {user_message}
AI response: {ai_response}

Response:"""


# ------------------------------------------------------------------- protocol

@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for local LLM inference. Both methods must be implemented."""

    def complete(self, prompt: str) -> str:
        """Send a prompt, return the completion text (stripped)."""
        ...


# ---------------------------------------------------------------- Ollama backend

class OllamaBackend:
    """Calls the Ollama REST API (http://localhost:11434 by default).

    Uses /api/generate with stream=false. Compatible with Ollama >= 0.1.
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG) -> None:
        self._base_url = cfg.LLM_BASE_URL.rstrip("/")
        self._model = cfg.LLM_MODEL
        self._timeout = cfg.LLM_TIMEOUT_SECONDS

    def complete(self, prompt: str) -> str:
        try:
            import requests
        except ImportError:
            raise RuntimeError(
                "OllamaBackend requires the 'requests' package. "
                "Install it: pip install requests"
            )
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},  # deterministic extraction
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.error("OllamaBackend.complete failed: %s", exc)
            raise


# ---------------------------------------------------------- response parsers

def parse_features(raw: str) -> list[str]:
    """Parse the LLM's feature extraction response into a list of strings."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x).strip().lower() for x in result if x]
    except json.JSONDecodeError:
        pass
    # Fallback: extract quoted strings from the raw output.
    return re.findall(r'"([^"]+)"', raw)[:5] or ["query"]


def parse_grains(raw: str) -> list[dict]:
    """Parse the LLM's grain extraction response into a list of grain dicts."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            valid = []
            for item in result:
                if isinstance(item, dict) and "content" in item:
                    provenance = item.get("provenance", "ai_stated")
                    if provenance not in ("user_stated", "ai_stated", "ai_inferred", "external_source"):
                        provenance = "ai_stated"
                    valid.append({"content": str(item["content"]).strip(), "provenance": provenance})
            return valid
    except json.JSONDecodeError:
        pass
    return []
