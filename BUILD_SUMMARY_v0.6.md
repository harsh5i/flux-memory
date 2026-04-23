# Flux Memory v0.6 Build Summary

## Overview

v0.6 introduces the first packaged Flux Memory release with a CLI, REST API,
dashboard, booth-style service wrapper, admin authentication, and stdio MCP
server support. Post-release Windows validation found several install and
runtime issues that must be treated as release blockers before calling the
package production-ready.

## What Changed

### New components

| File | Description |
|------|-------------|
| `src/flux/service.py` | `FluxService` booth architecture with read workers, serial writes, async feedback, and per-caller rate limiting |
| `src/flux/rest_api.py` | FastAPI REST API for store, retrieve, feedback, health, and grain listing |
| `src/flux/admin_auth.py` | Admin password hashing, TOTP support, lockout, and session tokens |
| `src/flux/cli.py` | Click CLI for `init`, `start`, `stop`, `status`, `admin`, `mcp`, and `mcp-config` |
| `src/flux/__main__.py` | `python -m flux` fallback for environments where the `flux` console script is not on PATH |

### Corrected post-release issues

| Issue | Fix |
|-------|-----|
| Windows `pip install --user` can install `flux.exe` outside PATH | Documented `python -m flux` fallback and recommended `pipx`; added package module entry point |
| `flux start` claimed success before services were reachable | Start now writes logs, waits for REST/dashboard health, and reports partial failure with log path |
| Dashboard API could hang because the dashboard used single-threaded `HTTPServer` | Dashboard now uses `ThreadingHTTPServer` |
| Dashboard could fail under Windows console encodings due Unicode status output | Console startup message now uses ASCII |
| REST root returned 404 | Added `GET /` service metadata endpoint |
| REST bound to `0.0.0.0` by default | Added host config defaults and bound local services to `127.0.0.1` |
| MCP server was stdio-only but `flux start` implied it was online/discoverable | Added explicit `flux mcp --name <instance>` command and generated client snippets |
| MCP dependency was not declared | Added `mcp>=1.0` to default dependencies |
| TOTP QR required optional dependency not installed by default | Added `qrcode>=7.0` to default dependencies |
| TOTP setup did not require first-code verification | Init now verifies a 6-digit TOTP code or disables TOTP |
| Project URLs pointed to the old repo name | Updated metadata and docs to `harsh5i/flux-memory` |

### Existing core changes from v0.6

| File | Change |
|------|--------|
| `src/flux/mcp_server.py` | Server name from `cfg.MCP_SERVER_NAME`; added `flux_onboard` and `flux_list_grains`; caller IDs on tools; optional service dispatch |
| `src/flux/config.py` | Added operating mode, service ports/hosts, booth tuning params, and admin auth params |
| `src/flux/storage.py` | `check_same_thread=False` for multi-threaded access and `INSERT OR IGNORE` for concurrent entry creation |
| `src/flux/llm.py` / `src/flux/embedding.py` | Moved test mocks out of production modules |
| `pyproject.toml` | Version 0.6.1 package metadata and console script |

## Validation

At the original v0.6 release, the suite was reported as:

```text
399 passed, 0 failed, 0 skipped
```

Post-release validation for the Windows install/runtime fixes included:

```text
python -m compileall src\flux
python -m flux --help
python -m flux mcp-config --name test1
direct REST root smoke check
direct TOTP verify/disable smoke check
direct MCP snippet generation smoke check
```

The local machine used for this follow-up could not run pytest because pytest
failed before executing project code with `PermissionError: [WinError 5]` while
creating its temp directory. Treat the direct smoke checks as patch validation,
not as a replacement for CI or a clean-machine pytest run.

## Remaining Release Notes

- `pip install flux-memory` itself is controlled by pip and will remain plain
  text. Flux can improve the first-run CLI experience, but package install
  output should not rely on post-install scripts.
- Stdio MCP servers are not automatically discoverable by Codex, Claude, or
  Cursor. Users must add the generated client config snippet, or the product
  must ship an explicit installer/integration command for each client.
- The dashboard and MCP flows need end-to-end CI coverage on Windows before the
  release should be described as production-ready.

## Installation

```bash
pip install flux-memory==0.6.1
python -m flux init --name my-memory
python -m flux start --name my-memory
python -m flux mcp-config --name my-memory
```

## Repository

https://github.com/harsh5i/flux-memory

Tag: `v0.6`
