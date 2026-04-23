# Flux Memory

**Self-organizing retrieval fabric for AI memory.**

Flux Memory is an AI memory system that persists knowledge as a self-modifying weighted graph. It learns which memories matter through feedback signals - reinforcing useful grains, decaying stale ones, and automatically clustering related knowledge.

## Features

- **Graph-based memory** - grains connected by weighted conduits, propagated via signal attenuation
- **Self-organizing** - lazy decay, Louvain clustering, automatic promotion/demotion, shortcut reinforcement
- **Three access paths** - MCP server (for AI agents), REST API (HTTP), Python SDK
- **Booth architecture** - concurrent read workers, serial write queue, async feedback queue
- **Per-caller rate limiting** - 500 grains/min default, configurable per instance
- **Admin authentication** - argon2 password hashing, TOTP 2FA (RFC 6238), session tokens
- **Two operating modes** - `flux_extracts` (local Ollama LLM) or `caller_extracts` (AI provides features)

## Quick Start

### Install

```bash
pip install flux-memory
```

Windows fallback if `flux` is not on PATH:

```bat
python -m flux --help
python -m flux init --name my-memory
```

For CLI-first installs, `pipx install flux-memory` is recommended because it manages command shims and PATH setup.

### Initialize an instance

```bash
flux init --name my-memory
```

This prompts for:
- Operating mode (`caller_extracts` or `flux_extracts`)
- Admin password (argon2-hashed)
- Optional TOTP two-factor authentication, with terminal QR and first-code verification

Initialization also writes MCP client snippets under:

```text
~/.flux/<name>/integrations/
```

### Start services

```bash
flux start --name my-memory
```

Starts:
- REST API health endpoint at `http://localhost:7465/health`
- Dashboard at `http://localhost:7462`

`flux start` does not make the stdio MCP server discoverable by itself. MCP clients launch stdio servers directly. Use the generated snippet or run:

```bash
flux mcp --name my-memory
```

from your MCP client configuration.

### Stop services

```bash
flux stop --name my-memory
```

### Check status

```bash
flux status --name my-memory
```

## MCP Integration

Connect Flux Memory to any MCP-compatible AI agent. Flux uses stdio MCP by default, so the client must launch Flux.

Generate or refresh client snippets:

```bash
flux mcp-config --name my-memory
```

Codex example:

```toml
[mcp_servers."flux-my-memory"]
command = "python"
args = ["-m", "flux.cli", "mcp", "--name", "my-memory"]
```

On first connection, call `flux_onboard` to receive integration instructions:

```
flux_onboard() -> returns workflow instructions + operating mode
```

**Standard workflow per conversation turn:**

1. `flux_retrieve(query)` - fetch relevant memories before responding
2. `flux_store(content, provenance)` - save new facts after responding
3. `flux_feedback(trace_id, grain_id, useful)` - rate each retrieved grain

**Available MCP tools:**

| Tool | Description |
|------|-------------|
| `flux_store` | Store a memory grain |
| `flux_retrieve` | Retrieve relevant memories |
| `flux_feedback` | Rate a retrieved grain (learning signal) |
| `flux_health` | Current health and signal statistics |
| `flux_list_grains` | List grains by status (active/dormant/quarantined/archived) |
| `flux_onboard` | Get integration instructions for this instance |

## REST API

```http
POST /store          {"content": "...", "provenance": "user_stated"}
POST /store/batch    {"items": [{"content": "..."}]}
POST /retrieve       {"query": "..."}
POST /feedback       {"trace_id": "...", "grain_id": "...", "useful": true}
GET  /health
GET  /grains?status=active&limit=50
```

Pass `X-Caller-Id: agent-name` header for per-caller rate limiting and tracking.

## Python SDK

```python
from flux.storage import FluxStore
from flux.service import FluxService
from flux.config import Config

store = FluxStore("~/.flux/my-memory/flux.db")
svc = FluxService(store, cfg=Config())
svc.start()

grain_id = svc.store("Paris is the capital of France", provenance="user_stated")
result = svc.retrieve("French capital")
svc.feedback(result.trace_id, result.grains[0]["id"], useful=True)

svc.stop()
store.close()
```

## Configuration

Instance config lives at `~/.flux/<name>/config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPERATING_MODE` | `flux_extracts` | LLM extraction mode |
| `MCP_HOST` | `127.0.0.1` | Reserved for network MCP transports |
| `MCP_PORT` | `7464` | Reserved MCP port |
| `REST_HOST` | `127.0.0.1` | REST bind host |
| `REST_PORT` | `7465` | REST API port |
| `DASHBOARD_HOST` | `127.0.0.1` | Dashboard bind host |
| `DASHBOARD_PORT` | `7462` | Dashboard port |
| `READ_WORKERS` | `3` | Concurrent read workers |
| `MAX_GRAINS_PER_CALL` | `100` | Batch ingestion cap |
| `MAX_GRAINS_PER_MINUTE` | `500` | Per-caller rate limit |
| `MAX_WRITE_QUEUE_DEPTH` | `1000` | Write queue backpressure cap |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama model (flux_extracts mode) |

## Admin

```bash
flux admin --name my-memory
```

Password + TOTP gated interactive menu: search/purge/restore grains, view audit log, change password, open dashboard.

## Requirements

- Python 3.10+
- SQLite 3.35+ (WAL mode)
- For `flux_extracts` mode: [Ollama](https://ollama.ai) with `qwen2.5:7b-instruct`

## Development

```bash
git clone https://github.com/harsh5i/flux-memory
cd flux-memory
pip install -e ".[test]"
pytest tests/
```

## License

MIT - see [LICENSE](LICENSE)
