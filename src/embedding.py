"""
Embedding Bootstrap — Use embeddings to create better initial conduits.

When a new grain is added, we can use semantic similarity to connect it
to existing grains, creating better initial paths than keyword matching.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import hashlib


class EmbeddingBootstrap:
    """
    Use embeddings to bootstrap conduits for new grains.
    
    Integrates with OpenClaw's embedding system (embeddinggemma-300m, 768-dim)
    or can use Ollama for local embeddings.
    """
    
    def __init__(
        self,
        api_base: str = "http://127.0.0.1:11434",
        embedding_model: str = "nomic-embed-text",  # Ollama default
    ):
        self.api_base = api_base
        self.embedding_model = embedding_model
        self._cache: Dict[str, List[float]] = {}
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text via Ollama API."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            import requests
            
            response = requests.post(
                f"{self.api_base}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text[:2000],  # Ollama uses 'prompt', not 'input'
                },
                timeout=10,
            )
            response.raise_for_status()
            
            embedding = response.json()["embedding"]  # Ollama returns direct embedding
            self._cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            # Fallback: Return None to signal embedding unavailable
            # Caller should use keyword-based matching instead
            print(f"[Flux] Embedding failed ({self.embedding_model}): {e}")
            print(f"[Flux] Falling back to keyword-based matching")
            return None
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        
        if mag_a == 0 or mag_b == 0:
            return 0.0
        
        return dot / (mag_a * mag_b)
    
    def find_similar_grains(
        self,
        new_grain_content: str,
        existing_grains: Dict[str, "Grain"],
        threshold: float = 0.3,
        max_connections: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find grains similar to new grain content.
        
        Returns list of (grain_id, similarity) tuples.
        """
        if not existing_grains:
            return []
        
        new_embedding = self.get_embedding(new_grain_content)
        if not new_embedding:
            return []
        
        similarities = []
        for gid, grain in existing_grains.items():
            grain_embedding = self.get_embedding(grain.content)
            if grain_embedding:
                sim = self.cosine_similarity(new_embedding, grain_embedding)
                if sim >= threshold:
                    similarities.append((gid, sim))
        
        # Sort by similarity, top N
        similarities.sort(key=lambda x: -x[1])
        return similarities[:max_connections]
    
    def bootstrap_conduits(
        self,
        new_grain: "Grain",
        existing_grains: Dict[str, "Grain"],
        min_similarity: float = 0.3,
        max_connections: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Create bootstrap conduits for a new grain.
        
        Returns list of (target_grain_id, weight) for conduits to create.
        """
        # Filter out self from existing grains
        filtered_grains = {
            gid: grain for gid, grain in existing_grains.items()
            if gid != new_grain.id
        }
        
        similar = self.find_similar_grains(
            new_grain.content,
            filtered_grains,  # Pass filtered grains
            threshold=min_similarity,
            max_connections=max_connections,
        )
        
        # Map similarity to conduit weight
        # similarity 0.3 → weight 0.2
        # similarity 0.7 → weight 0.5
        # similarity 1.0 → weight 0.7
        result = []
        for gid, sim in similar:
            weight = 0.2 + (sim * 0.5)  # Scale to [0.2, 0.7]
            result.append((gid, weight))
        
        return result


def bootstrap_from_embeddings(
    new_grain: "Grain",
    existing_grains: Dict[str, "Grain"],
    embedding_model: str = "nomic-embed-text",
) -> List[Tuple[str, float]]:
    """
    Convenience function for embedding-based bootstrap.
    """
    bootstrapper = EmbeddingBootstrap(embedding_model=embedding_model)
    return bootstrapper.bootstrap_conduits(new_grain, existing_grains)