# Flux Memory v0.6 — Build Summary

## Overview

v0.6 is a production-ready, deployable release. It adds a full CLI, REST API, booth architecture, admin authentication, and MCP onboarding on top of the v0.5 engine.

## What Changed

### New components

| File | Description |
|------|-------------|
| `src/flux/service.py` | `FluxService` booth architecture — ThreadPoolExecutor for reads, serial Queue for writes, async Queue for feedback, per-caller sliding-window rate limiting |
| `src/flux/rest_api.py` | FastAPI REST API — POST /store /store/batch /retrieve /feedback, GET /health /grains, X-Caller-Id header |
| `src/flux/admin_auth.py` | argon2 password hashing, pyotp TOTP (RFC 6238), 3-attempt lockout, session tokens |
| `src/flux/cli.py` | click CLI — `flux init/start/stop/status/admin`, background daemon via subprocess.Popen, PID file, interactive admin menu |

### Modified components

| File | Change |
|------|--------|
| `src/flux/mcp_server.py` | Server name from `cfg.MCP_SERVER_NAME`; added `flux_onboard` and `flux_list_grains` tools; `caller_id` on all tools; booth-aware dispatch via optional `service=` |
| `src/flux/config.py` | Added `OPERATING_MODE`, `MCP_SERVER_NAME`, port fields, booth tuning params (`READ_WORKERS`, `MAX_GRAINS_PER_CALL`, `MAX_WRITE_QUEUE_DEPTH`, `MAX_GRAINS_PER_MINUTE`), admin auth params; LLM default → `qwen2.5:7b-instruct` |
| `src/flux/storage.py` | `check_same_thread=False` for multi-threaded access; `INSERT OR IGNORE` on `entries.feature` for concurrent-retrieve race safety |
| `src/flux/llm.py` | Removed `MockLLMBackend` (moved to `tests/mocks.py`) |
| `src/flux/embedding.py` | Removed `MockEmbeddingBackend` (moved to `tests/mocks.py`) |
| `src/flux/__init__.py` | Removed mock exports from public API |
| `pyproject.toml` | Version 0.6.0; new deps: fastapi, uvicorn, click, argon2-cffi, pyotp; `flux` CLI entry point; MIT license; project URLs |

### Bug fixes

| Bug | Fix |
|-----|-----|
| §1A.9: `count_inbound_conduits` missed bidirectional shortcuts | SQL now checks `to_id=grain_id OR (from_id=grain_id AND direction='bidirectional')` |
| §1B.1: mocks in production code | Moved `MockLLMBackend` and `MockEmbeddingBackend` to `tests/mocks.py` |
| SQLite cross-thread crash in booth | Added `check_same_thread=False` to `FluxStore` connection |
| Concurrent retrieval UNIQUE constraint race | `INSERT OR IGNORE` on `entries.feature` |
| FastAPI annotation resolution with local Pydantic models | Moved models to module scope in `rest_api.py` |

## Test Results

```
399 passed, 0 failed, 0 skipped
```

### New test files (89 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_service.py` | 25 | Booth: rate limiter, lifecycle, store/batch/retrieve/feedback, concurrency |
| `tests/test_rest_api.py` | 26 | REST: all endpoints, caller ID, rate limits, batch cap, status filters |
| `tests/test_admin_auth.py` | 22 | Auth: setup, authenticate, lockout, sessions, password change, TOTP |
| `tests/test_mcp_v6.py` | 16 | MCP: onboarding, list_grains, caller_id, booth dispatch, unknown tool |

### Existing tests retained: 310 (zero regressions)

### §1B.5 compliance test

`tests/test_decay.py::TestBidirectionalOrphanFix` — two tests verify:
1. A grain reachable only via a bidirectional shortcut (`from_id=grain_id`) is NOT incorrectly marked dormant
2. A grain with only a forward conduit still decays correctly (regression guard)

## Spec compliance

| Requirement | Status |
|-------------|--------|
| §1A.1 `flux init` CLI | ✅ |
| §1A.2 `flux start/stop/status/admin` | ✅ |
| §1A.4 REST API | ✅ |
| §1A.5 `flux_onboard` MCP tool | ✅ |
| §1A.6 `caller_id` on all MCP/REST | ✅ |
| §1A.7 Booth architecture | ✅ |
| §1A.7a Ingestion limits | ✅ |
| §1A.8 Admin auth (argon2 + TOTP) | ✅ |
| §1A.9 Bidirectional orphan fix | ✅ |
| §1B.1 Zero mocks in production code | ✅ |
| §1B.4 399 tests, zero skips | ✅ |
| §1B.5 Bidirectional orphan test | ✅ |

## Installation

```bash
pip install flux-memory==0.6.0
flux init --name my-memory
flux start --name my-memory
```

## Repository

https://github.com/harsh5i/v0.5-flux-memory

Tag: `v0.6`
