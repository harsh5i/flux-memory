# Flux Memory

A self-organizing retrieval fabric for AI memory.

**Core principle:** Retrieval is not search — it's signal propagation through a self-modifying weighted graph. Every retrieval reshapes the ability to find again.

## Status

🚧 In Development

## Architecture

```
src/
├── grain.py          # Atomic memory unit
├── conduit.py        # Weighted edge between grains
├── entry_point.py    # Query feature → graph interface
├── signal.py         # Propagation engine
├── trace.py          # Retrieval receipt
├── store.py          # Persistence layer
├── decay.py          # Time-based weight/pruning
└── flux.py           # Main API
```

## Quick Start

```bash
# TODO
```

## Key Concepts

### Grain (G)
Atomic memory item. Immutable. Has `id`, `content`, `decay_class` (working|core), `context_spread`.

### Conduit (C)
Directional weighted edge between grains. Properties: `weight`, `last_used`, `use_count`. Strengthened/weakened through use.

### Entry Point (E)
Where query signal enters the fabric. Created from query decomposition. Has learned affinities toward proven first-hop conduits.

### Trace (T)
Recorded path of signal during retrieval. Used for weight updates.

## Retrieval Flow

1. Query → LLM decomposition → features
2. Features → Entry Points
3. Signal (1.0) injected at entry points
4. Propagate through conduits (attenuate per hop)
5. Collect grains above threshold
6. Update weights based on trace
7. Return results

## Decay

| Class | Half-Life | When Assigned |
|-------|-----------|---------------|
| Core | 720h (~30d) | Grains with context_spread >= 3 |
| Working | 168h (~7d) | Default for new grains |
| Ephemeral | 48h (~2d) | Session-specific (optional) |

## License

MIT