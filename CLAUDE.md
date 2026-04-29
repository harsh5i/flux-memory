# Flux Memory v0.5 — CLAUDE.md

Spec: `docs/flux-memory-docs.md`  
Checkpoint log: keep local checkpoint files outside this repository.

## Hard constraints (never deviate)
- Python 3.10+, SQLite 3.35+ with WAL mode
- `networkx.community.louvain_communities` for clustering — NOT python-louvain
- Lazy decay only (effective_weight computed at read time from last_used)
- MAX_EDGES_PER_GRAIN cap with weakest-edge eviction
- Every grain has provenance: user_stated / ai_stated / ai_inferred / external_source
- Admin channel gatekeeper protocol (§7.6): confirmation token, rate limiting, audit log, 24h purge undo window. Not exposed via MCP.
- Soft cluster membership with touch-weight accumulator and split/merge remapping (§13.2). NOT disjoint.
- Multi-signal feedback with provenance multipliers (§7.1). NOT single boolean.
- Health Monitor is first-class — build alongside the engine, not after.

## Repo layout
```
src/flux/
  __init__.py       public API
  config.py         all parameters (Section 5)
  graph.py          data classes
  storage.py        SQLite access layer
  propagation.py    signal propagation + lazy decay
  reinforcement.py  reinforce + penalize + shortcuts
  decay.py          cleanup_pass + expiry_pass
  clustering.py     Louvain soft clustering
  promotion.py      grain promotion
  [health.py]       Health Monitor (Track 3)
  [extraction.py]   grain extractor LLM (Track 2)
  [retrieval.py]    high-level flux_retrieve surface (Track 4)
  [admin.py]        admin channel (Track 4)
tests/
  conftest.py
  test_config.py
  test_schema.py
  test_storage.py
  test_propagation.py
  test_reinforcement.py
  test_decay.py
  test_clustering.py
  test_promotion.py
```

## Development rules
- One algorithm at a time, with pytest tests before moving on.
- Commit at each Track 1 step boundary with message: `Track 1 step N: description (§X.Y)`
- Spec unclear? STOP and note the question. Do not silently improvise.
- Internal helpers not in the spec: implement simplest version consistent with usage and note the choice.
- Structured JSON logging (§11.5).
- All parameters from config.py — none hardcoded.

## Track status (see CHECKPOINT.md for full detail)
- Track 1 steps 1–8: ✅ complete
- Track 2–6: ⬜ not started
