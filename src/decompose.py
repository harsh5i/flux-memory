"""
Query Decomposition — Extract features from queries.

Uses LLM to decompose queries into meaningful features for entry points.
"""

import json
from typing import List, Callable, Optional
import os


DECOMPOSITION_PROMPT = """Extract key features/concepts from this query for memory retrieval.

Query: {query}

Return ONLY a JSON array of 2-5 keywords or short phrases that capture the main concepts.
Focus on nouns, named entities, and key terms — not common words.

Examples:
Query: "What's the VMO2 project deadline?"
Output: ["VMO2", "project", "deadline"]

Query: "How do I configure the Dhan trading strategy?"
Output: ["Dhan", "trading", "configure", "strategy"]

Now process this query and return ONLY the JSON array:
"""


def extract_features_simple(query: str) -> List[str]:
    """
    Simple keyword extraction (fallback if no LLM available).
    """
    # Common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "what", "when", "where",
        "which", "how", "why", "who", "do", "does", "did", "can", "could",
        "would", "should", "will", "be", "been", "being", "have", "has", "had",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "not", "only", "own", "same", "than", "too", "very", "just", "also",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their", "this", "that",
        "these", "those", "am", "about", "if", "then", "else", "all", "any",
    }
    
    words = query.lower().split()
    # Clean punctuation
    words = [w.strip(".,!?;:'\"()[]{}") for w in words]
    # Filter
    features = [w for w in words if len(w) > 2 and w not in stop_words]
    # Dedupe, limit
    return list(dict.fromkeys(features))[:5]


def extract_features_llm(
    query: str,
    model: str = "glm-5.1:cloud",
    api_base: str = "http://127.0.0.1:11434/v1",
) -> List[str]:
    """
    Use LLM to extract features from query.
    
    Requires Ollama running locally with the specified model.
    """
    try:
        import requests
        
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        
        response = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a query analyzer. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.1,
            },
            timeout=15,
        )
        response.raise_for_status()
        
        resp_json = response.json()
        content = resp_json["choices"][0]["message"].get("content", "").strip()
        
        # If content is empty, check reasoning field (some models put output there)
        if not content:
            reasoning = resp_json["choices"][0]["message"].get("reasoning", "").strip()
            if reasoning:
                # Try to extract JSON array from reasoning
                import re
                match = re.search(r'\[.*?\]', reasoning, re.DOTALL)
                if match:
                    content = match.group(0)
        
        # Parse JSON array
        # Handle markdown code blocks if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip("`\n ")
        
        features = json.loads(content)
        
        if isinstance(features, list) and all(isinstance(f, str) for f in features):
            return features[:5]
        
    except Exception as e:
        print(f"[Flux] LLM decomposition failed: {e}")
    
    # Fallback to simple extraction
    return extract_features_simple(query)


class QueryDecomposer:
    """
    Query decomposition with LLM fallback and query expansion.
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        model: str = None,
        api_base: str = "http://127.0.0.1:11434/v1",
        expand_query: bool = True,
    ):
        self.use_llm = use_llm
        self.model = model or os.environ.get("FLUX_DECOMPOSE_MODEL", "gemma4:e2b")
        self.api_base = api_base
        self.expand_query = expand_query
    
    def decompose(self, query: str) -> List[str]:
        """Extract features from a query, with optional expansion."""
        if self.use_llm:
            features = extract_features_llm(query, self.model, self.api_base)
        else:
            features = extract_features_simple(query)
        
        if self.expand_query and features:
            expanded = self._expand_features(features, query)
            existing = set(f.lower() for f in features)
            for exp in expanded:
                if exp.lower() not in existing:
                    features.append(exp)
                    existing.add(exp.lower())
        
        return features[:10]  # Cap at 10 to prevent explosion
    
    def _expand_features(self, features: List[str], original_query: str) -> List[str]:
        """Generate related search terms using LLM. Falls back to synonyms if LLM fails."""
        if self.use_llm:
            try:
                import requests
                import re
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Given search features, return 3-5 RELATED features "
                                    "that would help find relevant information. "
                                    "Return ONLY a JSON array. No explanation."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Features: {json.dumps(features)}\nQuery: {original_query}"
                            }
                        ],
                        "max_tokens": 200,
                        "temperature": 0.3,
                    },
                    timeout=8,
                )
                response.raise_for_status()
                
                resp_json = response.json()
                content = resp_json["choices"][0]["message"].get("content", "").strip()
                
                if not content:
                    reasoning = resp_json["choices"][0]["message"].get("reasoning", "").strip()
                    if reasoning:
                        match = re.search(r'\[.*?\]', reasoning, re.DOTALL)
                        if match:
                            content = match.group(0)
                
                if content:
                    if "```" in content:
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip("`\n ")
                    
                    expanded = json.loads(content)
                    if isinstance(expanded, list):
                        return [str(f).strip() for f in expanded if str(f).strip()]
            except Exception:
                pass
        
        # Fallback: simple synonym expansion
        synonym_map = {
            'fix': ['bug', 'repair'], 'bug': ['fix', 'error'],
            'improve': ['enhance', 'optimize'], 'learn': ['study', 'research'],
            'build': ['create', 'implement'], 'security': ['vulnerability', 'defense'],
            'trading': ['strategy', 'market'], 'memory': ['recall', 'storage'],
        }
        expanded = []
        for feature in features:
            fl = feature.lower()
            for key, syns in synonym_map.items():
                if key == fl:  # Only exact match, not substring
                    for syn in syns[:1]:
                        expanded.append(syn)
                    break
        return expanded[:5]
    
    def __call__(self, query: str) -> List[str]:
        return self.decompose(query)