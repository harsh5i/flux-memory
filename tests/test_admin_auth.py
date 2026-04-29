"""Tests for AdminAuth (§1A.8) — password, TOTP, lockout, sessions."""
from __future__ import annotations

import time

import pytest

from flux.admin_auth import AdminAuth


@pytest.fixture
def auth(tmp_path):
    return AdminAuth(
        tmp_path / "flux",
        lockout_minutes=1,
        max_attempts=3,
        session_hours=1,
    )


# ---------------------------------------------------------------- setup

class TestAdminAuthSetup:
    def test_not_configured_before_setup(self, auth):
        assert auth.is_configured() is False

    def test_configured_after_setup(self, auth):
        auth.setup("correct-horse", enable_totp=False)
        assert auth.is_configured() is True

    def test_setup_short_password_raises(self, auth):
        with pytest.raises(ValueError, match="8 characters"):
            auth.setup("short")

    def test_setup_persists_across_instances(self, tmp_path):
        d = tmp_path / "flux"
        a1 = AdminAuth(d)
        a1.setup("persistentpw", enable_totp=False)
        a2 = AdminAuth(d)
        assert a2.is_configured() is True

    def test_setup_without_totp_returns_none(self, auth):
        uri = auth.setup("validpassword", enable_totp=False)
        assert uri is None

    def test_setup_with_totp_returns_uri(self, auth):
        try:
            import pyotp  # noqa: F401
            uri = auth.setup("validpassword", enable_totp=True)
            assert uri is not None
            assert "otpauth://" in uri
        except ImportError:
            pytest.skip("pyotp not installed")


# ---------------------------------------------------------------- authenticate

class TestAdminAuthAuthenticate:
    def test_correct_password_returns_token(self, auth):
        auth.setup("goodpassword", enable_totp=False)
        token = auth.authenticate("goodpassword")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_wrong_password_raises(self, auth):
        auth.setup("goodpassword", enable_totp=False)
        with pytest.raises(PermissionError, match="Invalid password"):
            auth.authenticate("wrongpassword")

    def test_wrong_password_decrements_attempts(self, auth):
        auth.setup("goodpassword", enable_totp=False)
        with pytest.raises(PermissionError, match="2 attempt"):
            auth.authenticate("wrong")

    def test_lockout_after_max_attempts(self, auth):
        auth.setup("goodpassword", enable_totp=False)
        for _ in range(3):
            try:
                auth.authenticate("wrong")
            except PermissionError:
                pass
        with pytest.raises(PermissionError, match="locked"):
            auth.authenticate("goodpassword")

    def test_success_resets_failed_counter(self, auth):
        auth.setup("goodpassword", enable_totp=False)
        try:
            auth.authenticate("bad")
        except PermissionError:
            pass
        token = auth.authenticate("goodpassword")
        assert token  # not locked


# ---------------------------------------------------------------- session

class TestAdminAuthSession:
    def test_verify_valid_session(self, auth):
        auth.setup("password123", enable_totp=False)
        token = auth.authenticate("password123")
        auth.verify_session(token)  # must not raise

    def test_verify_invalid_token_raises(self, auth):
        auth.setup("password123", enable_totp=False)
        with pytest.raises(PermissionError, match="Invalid or expired"):
            auth.verify_session("bogus-token")

    def test_invalidated_session_raises(self, auth):
        auth.setup("password123", enable_totp=False)
        token = auth.authenticate("password123")
        auth.invalidate_session(token)
        with pytest.raises(PermissionError):
            auth.verify_session(token)

    def test_multiple_sessions_independent(self, auth):
        auth.setup("password123", enable_totp=False)
        t1 = auth.authenticate("password123")
        t2 = auth.authenticate("password123")
        auth.invalidate_session(t1)
        auth.verify_session(t2)  # t2 still valid


# ---------------------------------------------------------------- change password

class TestAdminAuthChangePassword:
    def test_change_password(self, auth):
        auth.setup("original123", enable_totp=False)
        auth.authenticate("original123")
        auth.change_password("newpassword1")
        token = auth.authenticate("newpassword1")
        assert token

    def test_old_password_fails_after_change(self, auth):
        auth.setup("original123", enable_totp=False)
        auth.change_password("brandnewpw")
        with pytest.raises(PermissionError):
            auth.authenticate("original123")

    def test_short_new_password_raises(self, auth):
        auth.setup("original123", enable_totp=False)
        with pytest.raises(ValueError):
            auth.change_password("tiny")


# ---------------------------------------------------------------- TOTP

class TestAdminAuthTOTP:
    def test_totp_required_when_enabled(self, auth):
        try:
            import pyotp  # noqa: F401
        except ImportError:
            pytest.skip("pyotp not installed")
        auth.setup("goodpassword", enable_totp=True)
        with pytest.raises(PermissionError, match="TOTP"):
            auth.authenticate("goodpassword", totp_code=None)

    def test_totp_invalid_code_raises(self, auth):
        try:
            import pyotp  # noqa: F401
        except ImportError:
            pytest.skip("pyotp not installed")
        auth.setup("goodpassword", enable_totp=True)
        with pytest.raises(PermissionError):
            auth.authenticate("goodpassword", totp_code="000000")

    def test_totp_valid_code_succeeds(self, auth):
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")
        uri = auth.setup("goodpassword", enable_totp=True)
        secret = uri.split("secret=")[1].split("&")[0]
        totp = pyotp.TOTP(secret)
        code = totp.now()
        token = auth.authenticate("goodpassword", totp_code=code)
        assert token

    def test_verify_totp_code(self, auth):
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")
        uri = auth.setup("goodpassword", enable_totp=True)
        secret = uri.split("secret=")[1].split("&")[0]
        assert auth.verify_totp_code(pyotp.TOTP(secret).now())
        assert not auth.verify_totp_code("000000")

    def test_disable_totp(self, auth):
        auth.setup("goodpassword", enable_totp=True)
        auth.disable_totp()
        assert auth.totp_uri() is None
        assert auth.authenticate("goodpassword")

    def test_totp_uri_not_none_when_enabled(self, auth):
        try:
            import pyotp  # noqa: F401
        except ImportError:
            pytest.skip("pyotp not installed")
        auth.setup("goodpassword", enable_totp=True)
        uri = auth.totp_uri("test-label")
        assert uri is not None

    def test_totp_uri_none_when_disabled(self, auth):
        auth.setup("goodpassword", enable_totp=False)
        assert auth.totp_uri() is None
