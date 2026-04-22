"""Flux Memory CLI (§1A.1, §1A.2, §1A.8).

Entry point: `flux` command, exposed via pyproject.toml [project.scripts].

Subcommands:
  flux init   [--name NAME] [--db PATH] [--mode MODE]
              Interactive setup: creates instance config, hashes admin password,
              optionally sets up TOTP 2FA.

  flux start  [--name NAME]
              Launch MCP server, REST API, and dashboard as background services.

  flux stop   [--name NAME]
              Gracefully stop all services for the named instance.

  flux status [--name NAME]
              Show running service status and basic health.

  flux admin  [--name NAME]
              Interactive admin menu (password + TOTP gated).

  flux --version
              Print version and exit.

Instance config lives at: ~/.flux/<name>/config.yaml
Admin auth lives at:      ~/.flux/<name>/admin_auth.json
PID file lives at:        ~/.flux/<name>/flux.pid
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

_VERSION = "0.6.0"
_DEFAULT_NAME = "flux-memory"
_FLUX_HOME = Path.home() / ".flux"


def _instance_dir(name: str) -> Path:
    return _FLUX_HOME / name


def _pid_file(name: str) -> Path:
    return _instance_dir(name) / "flux.pid"


def _config_file(name: str) -> Path:
    return _instance_dir(name) / "config.yaml"


def _db_file(name: str, db: str | None = None) -> Path:
    if db:
        return Path(db).expanduser()
    return _instance_dir(name) / "flux.db"


# ---------------------------------------------------------------- main entry

def main() -> None:
    """Entry point for the `flux` command."""
    try:
        import click
    except ImportError:
        print("ERROR: 'click' package required. pip install flux-memory[cli]")
        sys.exit(1)
    cli()


def _require_click():
    try:
        import click
        return click
    except ImportError:
        print("ERROR: 'click' package required. pip install click")
        sys.exit(1)


# We import click lazily so the module can be imported without it.
try:
    import click

    @click.group()
    @click.version_option(version=_VERSION, prog_name="flux")
    def cli():
        """Flux Memory — self-organizing retrieval fabric for AI memory."""

    # ---------------------------------------------------------------- init

    @cli.command()
    @click.option("--name", default=_DEFAULT_NAME, show_default=True,
                  help="Instance name (used as MCP server name and directory label).")
    @click.option("--db", default=None, help="Path to SQLite database file.")
    @click.option("--mode", type=click.Choice(["caller_extracts", "flux_extracts"]),
                  default=None, help="Operating mode (prompted if omitted).")
    def init(name: str, db: str | None, mode: str | None) -> None:
        """Initialize a new Flux Memory instance."""
        idir = _instance_dir(name)
        idir.mkdir(parents=True, exist_ok=True)
        db_path = _db_file(name, db)

        click.echo(f"\n{'='*60}")
        click.echo(f"  Flux Memory — Instance Setup: {name}")
        click.echo(f"{'='*60}\n")

        # Operating mode.
        if mode is None:
            click.echo("Choose operating mode:")
            click.echo("  1) caller_extracts  — AI does feature/grain extraction (no local LLM needed)")
            click.echo("  2) flux_extracts    — Flux runs its own LLM via Ollama (requires Ollama)\n")
            choice = click.prompt("Mode", type=click.Choice(["1", "2"]), default="2")
            mode = "caller_extracts" if choice == "1" else "flux_extracts"

        # Write config.
        config = {
            "MCP_SERVER_NAME": name,
            "OPERATING_MODE": mode,
        }
        cfg_path = _config_file(name)
        import yaml
        cfg_path.write_text(yaml.dump(config, default_flow_style=False))

        click.echo(f"\nConfig written to: {cfg_path}")
        click.echo(f"Database will be:  {db_path}")

        # Admin password setup.
        click.echo("\n--- Admin password setup ---")
        click.echo("This password gates all destructive admin operations.")
        while True:
            pw = click.prompt("Admin password", hide_input=True)
            pw2 = click.prompt("Confirm password", hide_input=True)
            if pw != pw2:
                click.echo("Passwords do not match. Try again.")
                continue
            if len(pw) < 8:
                click.echo("Password must be at least 8 characters.")
                continue
            break

        enable_totp = click.confirm("\nEnable two-factor authentication (TOTP)?", default=True)

        from flux.admin_auth import AdminAuth
        auth = AdminAuth(
            idir,
            lockout_minutes=15,
            max_attempts=3,
            session_hours=1,
        )
        totp_uri = auth.setup(pw, enable_totp=enable_totp)

        if totp_uri:
            click.echo("\n--- TOTP Setup ---")
            click.echo("Scan the QR code below with your authenticator app")
            click.echo("(Google Authenticator, Authy, 1Password, etc.):\n")
            auth.show_qr()
            click.echo(f"\nTOTP URI (if QR display failed):\n{totp_uri}\n")
            click.confirm("Have you scanned the QR code?", default=True)

        click.echo(f"\n✓ Instance '{name}' initialized.")
        click.echo(f"  Run `flux start --name {name}` to launch services.")

    # ---------------------------------------------------------------- start

    @cli.command()
    @click.option("--name", default=_DEFAULT_NAME, show_default=True)
    @click.option("--foreground", is_flag=True, default=False,
                  help="Run in foreground (blocking). Default: background.")
    def start(name: str, foreground: bool) -> None:
        """Launch MCP server, REST API, and dashboard."""
        idir = _instance_dir(name)
        if not _config_file(name).exists():
            click.echo(f"Instance '{name}' not initialized. Run: flux init --name {name}")
            sys.exit(1)

        pid_path = _pid_file(name)
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 0)
                click.echo(f"Instance '{name}' already running (PID {pid}).")
                return
            except OSError:
                pid_path.unlink(missing_ok=True)

        db_path = _db_file(name)
        cfg_path = _config_file(name)

        from flux.config import Config
        cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()

        click.echo(f"Starting Flux Memory instance '{name}'...")

        if foreground:
            _run_services(name, db_path, cfg)
        else:
            proc = subprocess.Popen(
                [sys.executable, "-m", "flux.cli", "start",
                 "--name", name, "--foreground"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            pid_path.write_text(str(proc.pid))
            click.echo(f"  Services started (PID {proc.pid})")
            click.echo(f"  REST API:  http://localhost:{cfg.REST_PORT}")
            click.echo(f"  Dashboard: http://localhost:{cfg.DASHBOARD_PORT}")
            click.echo(f"  Stop with: flux stop --name {name}")

    # ---------------------------------------------------------------- stop

    @cli.command()
    @click.option("--name", default=_DEFAULT_NAME, show_default=True)
    def stop(name: str) -> None:
        """Stop all services for the named instance."""
        pid_path = _pid_file(name)
        if not pid_path.exists():
            click.echo(f"No PID file found for instance '{name}'. May not be running.")
            return
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            pid_path.unlink(missing_ok=True)
            click.echo(f"Instance '{name}' (PID {pid}) stopped.")
        except OSError as exc:
            click.echo(f"Could not stop PID {pid}: {exc}")
            pid_path.unlink(missing_ok=True)

    # ---------------------------------------------------------------- status

    @cli.command()
    @click.option("--name", default=_DEFAULT_NAME, show_default=True)
    def status(name: str) -> None:
        """Show running status and health for the named instance."""
        pid_path = _pid_file(name)
        running = False
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 0)
                running = True
                click.echo(f"Instance '{name}': RUNNING (PID {pid})")
            except OSError:
                click.echo(f"Instance '{name}': STOPPED (stale PID file)")
                pid_path.unlink(missing_ok=True)
        else:
            click.echo(f"Instance '{name}': STOPPED")

        cfg_path = _config_file(name)
        if cfg_path.exists():
            from flux.config import Config
            cfg = Config.from_yaml(cfg_path)
            click.echo(f"  Mode:      {cfg.OPERATING_MODE}")
            click.echo(f"  REST port: {cfg.REST_PORT}")
            click.echo(f"  Dashboard: {cfg.DASHBOARD_PORT}")

        if running:
            try:
                import urllib.request
                from flux.config import Config
                cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()
                with urllib.request.urlopen(
                    f"http://localhost:{cfg.REST_PORT}/health", timeout=3
                ) as resp:
                    h = json.loads(resp.read())
                    click.echo(f"  Health:    {h.get('status', 'unknown')}")
            except Exception:
                click.echo("  Health:    (could not reach REST API)")

    # ---------------------------------------------------------------- admin

    @cli.command("admin")
    @click.option("--name", default=_DEFAULT_NAME, show_default=True)
    def admin_menu(name: str) -> None:
        """Interactive admin menu (password + TOTP gated)."""
        idir = _instance_dir(name)
        from flux.admin_auth import AdminAuth
        auth = AdminAuth(idir)

        if not auth.is_configured():
            click.echo(f"Instance '{name}' not initialized. Run: flux init --name {name}")
            sys.exit(1)

        click.echo(f"\n--- Flux Memory Admin: {name} ---")
        pw = click.prompt("Password", hide_input=True)
        totp_code = None
        # Check if TOTP enabled (we try with None first; auth will tell us if needed).
        try:
            token = auth.authenticate(pw, totp_code)
        except PermissionError as exc:
            msg = str(exc)
            if "TOTP" in msg:
                totp_code = click.prompt("TOTP code")
                try:
                    token = auth.authenticate(pw, totp_code)
                except PermissionError as exc2:
                    click.echo(f"Authentication failed: {exc2}")
                    sys.exit(1)
            else:
                click.echo(f"Authentication failed: {exc}")
                sys.exit(1)

        click.echo("\nAuthenticated.\n")

        db_path = _db_file(name)
        from flux.storage import FluxStore
        from flux.admin import flux_export_grain, flux_purge, flux_restore

        with FluxStore(db_path) as store:
            _admin_menu_loop(store, auth, token, name)

        auth.invalidate_session(token)

    def _admin_menu_loop(store, auth, token: str, name: str) -> None:
        """Interactive admin menu loop."""
        import os as _os
        MENU = """
Admin Menu:
  1) Search grains
  2) Purge a grain
  3) Restore a grain
  4) Export grain details
  5) View audit log
  6) Open dashboard
  7) Change password
  8) Exit
"""
        while True:
            click.echo(MENU)
            choice = click.prompt("Choice", type=click.Choice(["1","2","3","4","5","6","7","8"]))

            if choice == "1":
                pattern = click.prompt("Search pattern")
                rows = store.conn.execute(
                    "SELECT id, content, status, provenance FROM grains "
                    "WHERE content LIKE ? LIMIT 20",
                    (f"%{pattern}%",),
                ).fetchall()
                if not rows:
                    click.echo("No matches.")
                for r in rows:
                    click.echo(f"  [{r['status']}] {r['id'][:8]}.. {r['content'][:80]}")

            elif choice == "2":
                gid = click.prompt("Grain ID to purge")
                reason = click.prompt("Reason (required)")
                if click.confirm(f"Purge grain {gid}? This is irreversible.", default=False):
                    os.environ["FLUX_ADMIN_TOKEN"] = token
                    from flux.admin import flux_purge
                    result = flux_purge(gid, reason, store=store, confirmation_token=token)
                    click.echo(f"Purged. Conduits removed: {result['conduits_removed']}")

            elif choice == "3":
                gid = click.prompt("Grain ID to restore")
                os.environ["FLUX_ADMIN_TOKEN"] = token
                from flux.admin import flux_restore
                result = flux_restore(gid, store=store, confirmation_token=token)
                click.echo(f"Restored: {result['restored']} (was {result['previous_status']})")

            elif choice == "4":
                gid = click.prompt("Grain ID to inspect")
                os.environ["FLUX_ADMIN_TOKEN"] = token
                from flux.admin import flux_export_grain
                info = flux_export_grain(gid, store=store, confirmation_token=token)
                click.echo(json.dumps(info, indent=2, default=str))

            elif choice == "5":
                rows = store.conn.execute(
                    "SELECT timestamp, event, data FROM events "
                    "WHERE category='admin' ORDER BY timestamp DESC LIMIT 50"
                ).fetchall()
                for r in rows:
                    click.echo(f"  {r['timestamp']}  {r['event']}  {r['data'][:80]}")

            elif choice == "6":
                from flux.config import Config
                cfg_path = _config_file(name)
                cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()
                url = f"http://localhost:{cfg.DASHBOARD_PORT}"
                click.echo(f"Opening {url}")
                import webbrowser
                webbrowser.open(url)

            elif choice == "7":
                pw1 = click.prompt("New password", hide_input=True)
                pw2 = click.prompt("Confirm", hide_input=True)
                if pw1 != pw2:
                    click.echo("Passwords do not match.")
                else:
                    auth.change_password(pw1)
                    click.echo("Password changed.")

            elif choice == "8":
                break

    # ---------------------------------------------------------------- internal runner

    def _run_services(name: str, db_path: Path, cfg) -> None:
        """Start all services in the foreground (blocking)."""
        import threading
        from flux.storage import FluxStore
        from flux.service import FluxService

        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = FluxStore(db_path)
        service = FluxService(store, cfg=cfg)
        service.start()

        threads = []

        # REST API thread.
        try:
            import uvicorn
            from flux.rest_api import build_app
            app = build_app(service, cfg)

            def _rest():
                uvicorn.run(app, host="0.0.0.0", port=cfg.REST_PORT, log_level="warning")

            t = threading.Thread(target=_rest, name="flux-rest", daemon=True)
            t.start()
            threads.append(t)
        except ImportError:
            pass

        # Dashboard thread.
        try:
            from flux.dashboard import run_dashboard
            t = threading.Thread(
                target=run_dashboard,
                kwargs={"store": store, "cfg": cfg, "port": cfg.DASHBOARD_PORT},
                name="flux-dashboard",
                daemon=True,
            )
            t.start()
            threads.append(t)
        except Exception:
            pass

        # MCP server (blocking, must be last).
        try:
            from flux.mcp_server import run_stdio
            run_stdio(store, service._llm, service._emb, cfg)
        except KeyboardInterrupt:
            pass
        finally:
            service.stop()
            store.close()

except ImportError:
    # click not installed — provide a minimal shim so `python -m flux.cli` gives useful message.
    def main() -> None:  # type: ignore[misc]
        print("Flux Memory CLI requires 'click'. Install: pip install 'flux-memory[cli]'")
        sys.exit(1)

    def cli() -> None:  # type: ignore[misc]
        main()


if __name__ == "__main__":
    main()
