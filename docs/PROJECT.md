# Flux Memory — Project Spec

## Status
🚧 **In Development** — Core architecture complete, needs integration + testing

## Next Steps

### Immediate (Week 1)
- [ ] Run tests, fix any issues
- [ ] Add embedding-based bootstrap (use OpenClaw's existing embeddinggemma)
- [ ] Implement proper query decomposition via LLM
- [ ] Test with real queries

### Short-term (Week 2-3)
- [ ] Build MCP interface for use with OpenClaw
- [ ] Add co-retrieval edge creation (direct conduits between useful pairs)
- [ ] Implement entry point clustering (avoid fragmentation)
- [ ] Add trace analytics

### Medium-term (Month 1-2)
- [ ] Integrate with OpenClaw memory system
- [ ] Replace/augment MEMORY.md backend
- [ ] Add export/import for grains
- [ ] Performance optimization (batch operations, caching)

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Flux API                   │
│   remember() | query() | feedback() | ...   │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌────────────┐ ┌─────────┐ ┌───────────┐
│ Signal     │ │  Store  │ │  Decay    │
│ Engine     │ │ (SQLite)│ │  Engine   │
└────────────┘ └─────────┘ └───────────┘
        │           │           │
        ▼           ▼           ▼
┌─────────────────────────────────────────────┐
│            Data Structures                  │
│   Grain | Conduit | EntryPoint | Trace     │
└─────────────────────────────────────────────┘
```

## Dependencies

- Python 3.10+
- SQLite (built-in)
- Optional: OpenAI/Anthropic for query decomposition
- Optional: embedding model for bootstrap

## Key Design Decisions

1. **Grain immutability** — Content never changes after creation
2. **Conduit weights mutate** — Retrieval behavior drives learning
3. **Promotion through use** — context_spread >= 3 → core class
4. **SQLite for persistence** — Simple, fast, reliable
5. **Decay as explicit operation** — Run periodically, not per-query

## Open Questions

1. Entry point clustering — how to avoid fragmentation?
2. Co-retrieval threshold — how many co-occurrences before direct conduit?
3. Batch vs online updates — trade-off between consistency and latency
4. Trace archival policy — keep forever or summarize?

## Files

```
src/
├── grain.py          # ✓ Atomic memory unit
├── conduit.py        # ✓ Weighted edge
├── entry_point.py   # ✓ Query → graph interface
├── trace.py         # ✓ Retrieval receipt
├── signal.py        # ✓ Propagation engine
├── store.py         # ✓ SQLite persistence
├── decay.py         # ✓ Time-based pruning
├── flux.py          # ✓ Main API
└── __init__.py      # ✓ Package init

tests/
└── test_flux.py     # ✓ Unit tests

docs/
└── ...              # TODO: API docs
```