"""Admin authentication layer (§1A.8).

Password hashing: argon2-cffi (modern, memory-hard).
Two-factor:       pyotp (TOTP, RFC 6238 — works with any authenticator app).
Lockout:          3 failed attempts → ADMIN_LOCKOUT_MINUTES minute lockout.
Session tokens:   random hex token, expire in ADMIN_SESSION_HOURS.

All state is persisted in the flux instance config directory (~/.flux/<name>/).

Public API:
    auth = AdminAuth(config_dir)
    auth.setup(password, enable_totp=True)   # called from flux init
    auth.authenticate(password, totp_code)   # returns session token or raises
    auth.verify_session(token)               # raises if invalid/expired
    auth.totp_uri(label)                     # for QR code generation
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AUTH_FILE = "admin_auth.json"


def _hash_password(password: str) -> str:
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        return ph.hash(password)
    except ImportError:
        # Fallback to PBKDF2 if argon2-cffi not installed.
        salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
        return f"pbkdf2:{salt}:{key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("pbkdf2:"):
        _, salt, stored_key = stored_hash.split(":", 2)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
        return secrets.compare_digest(key.hex(), stored_key)
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError
        ph = PasswordHasher()
        try:
            ph.verify(stored_hash, password)
            return True
        except VerifyMismatchError:
            return False
    except ImportError:
        return False


class AdminAuth:
    """Manages admin credentials, lockout, and session tokens for one Flux instance."""

    def __init__(self, config_dir: Path, lockout_minutes: int = 15,
                 max_attempts: int = 3, session_hours: int = 1) -> None:
        self._dir = Path(config_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._auth_file = self._dir / _AUTH_FILE
        self._lockout_minutes = lockout_minutes
        self._max_attempts = max_attempts
        self._session_seconds = session_hours * 3600
        self._state: dict = self._load()

    # ---------------------------------------------------------------- setup

    def is_configured(self) -> bool:
        return bool(self._state.get("password_hash"))

    def setup(self, password: str, enable_totp: bool = False) -> str | None:
        """Hash and store the admin password. Returns TOTP URI if enabled."""
        if len(password) < 8:
            raise ValueError("Admin password must be at least 8 characters.")
        self._state["password_hash"] = _hash_password(password)
        self._state["totp_enabled"] = enable_totp
        self._state["failed_attempts"] = 0
        self._state["lockout_until"] = 0.0
        self._state["sessions"] = {}
        totp_uri = None
        if enable_totp:
            try:
                import pyotp
                secret = pyotp.random_base32()
                self._state["totp_secret"] = secret
                totp = pyotp.TOTP(secret)
                totp_uri = totp.provisioning_uri(name="flux-admin", issuer_name="Flux Memory")
            except ImportError:
                logger.warning("pyotp not installed — TOTP disabled. pip install pyotp")
                self._state["totp_enabled"] = False
        self._save()
        return totp_uri

    def change_password(self, new_password: str) -> None:
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        self._state["password_hash"] = _hash_password(new_password)
        self._save()

    # ---------------------------------------------------------------- authenticate

    def authenticate(self, password: str, totp_code: str | None = None) -> str:
        """Verify credentials and return a session token. Raises on failure."""
        now = time.time()
        if self._state.get("lockout_until", 0) > now:
            remaining = int((self._state["lockout_until"] - now) / 60) + 1
            raise PermissionError(
                f"Account locked. Try again in ~{remaining} minute(s)."
            )

        pw_ok = _verify_password(password, self._state.get("password_hash", ""))
        if not pw_ok:
            self._record_failure(now)
            attempts_left = self._max_attempts - self._state["failed_attempts"]
            if attempts_left <= 0:
                raise PermissionError("Account locked after too many failed attempts.")
            raise PermissionError(
                f"Invalid password. {attempts_left} attempt(s) remaining."
            )

        if self._state.get("totp_enabled"):
            if not totp_code:
                raise PermissionError("TOTP code required.")
            try:
                import pyotp
                totp = pyotp.TOTP(self._state["totp_secret"])
                if not totp.verify(totp_code, valid_window=1):
                    self._record_failure(now)
                    raise PermissionError("Invalid TOTP code.")
            except ImportError:
                logger.warning("pyotp not installed — skipping TOTP check")

        # Success — reset counter, issue token.
        self._state["failed_attempts"] = 0
        self._state["lockout_until"] = 0.0
        token = secrets.token_hex(32)
        sessions = self._state.setdefault("sessions", {})
        sessions[token] = now + self._session_seconds
        # Prune expired sessions.
        self._state["sessions"] = {
            t: exp for t, exp in sessions.items() if exp > now
        }
        self._save()
        return token

    def verify_session(self, token: str) -> None:
        """Raise PermissionError if token is invalid or expired."""
        sessions = self._state.get("sessions", {})
        expiry = sessions.get(token)
        if expiry is None or expiry < time.time():
            raise PermissionError("Invalid or expired admin session token.")

    def invalidate_session(self, token: str) -> None:
        sessions = self._state.get("sessions", {})
        sessions.pop(token, None)
        self._save()

    # ---------------------------------------------------------------- TOTP helpers

    def totp_uri(self, label: str = "flux-admin") -> str | None:
        if not self._state.get("totp_enabled"):
            return None
        try:
            import pyotp
            return pyotp.TOTP(self._state["totp_secret"]).provisioning_uri(
                name=label, issuer_name="Flux Memory"
            )
        except ImportError:
            return None

    def show_qr(self, label: str = "flux-admin") -> None:
        uri = self.totp_uri(label)
        if not uri:
            print("TOTP not enabled.")
            return
        try:
            import qrcode
            qr = qrcode.QRCode()
            qr.add_data(uri)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except ImportError:
            print(f"TOTP URI (scan with authenticator app):\n{uri}")
            print("(install 'qrcode' for inline QR: pip install qrcode)")

    # ---------------------------------------------------------------- internals

    def _record_failure(self, now: float) -> None:
        self._state["failed_attempts"] = self._state.get("failed_attempts", 0) + 1
        if self._state["failed_attempts"] >= self._max_attempts:
            self._state["lockout_until"] = now + self._lockout_minutes * 60
            logger.warning("Admin lockout triggered (%d min)", self._lockout_minutes)
        self._save()

    def _load(self) -> dict:
        if self._auth_file.exists():
            try:
                return json.loads(self._auth_file.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self._auth_file.write_text(json.dumps(self._state, indent=2))
        # Restrict permissions to owner only on Unix.
        try:
            os.chmod(self._auth_file, 0o600)
        except OSError:
            pass
