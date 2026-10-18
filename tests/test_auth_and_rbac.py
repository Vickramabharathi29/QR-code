"""Tests for Authentication, Role-Based Access Control (RBAC), and Security Enforcement.

Covers:
- Public access without login
- Successful administrator login
- Invalid credentials handling
- Public access denied for all 4 restricted sections
- Administrator access permitted after authentication
- Rejection of unauthorized users attempting admin elevation
- Logout and session clearance
- Supabase failure handling & missing environment variables
- Preservation of QR scanning and safe URL opening
- Protection against exposing another user's scan history
"""

from unittest.mock import MagicMock, patch
import pytest

from modules import auth, database
from modules.auth import (
    authenticate_admin,
    change_admin_password,
    get_admin_email,
    get_user_profile,
    provision_initial_admin,
    sign_out_admin,
    update_user_profile,
    validate_password_strength,
    verify_admin_authorization,
)
from modules.database import export_scan_history_to_csv, get_detailed_scan_history
from modules.threat_prevention import get_prevention_engine
from modules.url_analyzer import analyze_url


class TestPasswordValidation:
    """Test password complexity requirements for administrator accounts."""

    def test_weak_passwords_rejected(self):
        """Short, simple, or missing character types must be rejected."""
        assert validate_password_strength("short")[0] is False
        assert validate_password_strength("alllowercase123!")[0] is False
        assert validate_password_strength("ALLUPPERCASE123!")[0] is False
        assert validate_password_strength("NoSpecialChar123")[0] is False
        assert validate_password_strength("NoDigitsHere!@#")[0] is False

    def test_strong_password_accepted(self):
        """Complex passwords meeting standard criteria must be accepted."""
        valid, msg = validate_password_strength("Admin#QRShield2026!")
        assert valid is True
        assert "verified" in msg.lower()


class TestSupabaseAuthAndRBAC:
    """Test Supabase Auth administrator login, session verification, and RBAC."""

    def test_public_access_without_login(self):
        """Public users can analyze URLs and get safety decisions without login."""
        from modules.ml_classifier import predict_url
        engine = get_prevention_engine()
        url = "https://forms.gle/sampleSurvey123"
        analysis = analyze_url(url)
        ml_pred = predict_url(url)
        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTPS",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )
        assert nav["is_permitted"] is True
        assert nav["status_category"] == "PERMITTED"
        assert nav["button_label"] == "🌐 Open Google Form"

    def test_successful_admin_login_mocked(self):
        """Successful Supabase Auth returns authenticated admin session."""
        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "mock-admin-uuid-1234"
        mock_user.email = "vickramabharathi@gmail.com"
        mock_user.user_metadata = {"role": "admin", "full_name": "SecOps Administrator"}
        mock_user.app_metadata = {"role": "admin"}

        mock_session = MagicMock()
        mock_session.access_token = "mock-token-abc"
        mock_session.refresh_token = "mock-refresh-xyz"
        mock_session.expires_at = 1893456000

        mock_auth_resp = MagicMock()
        mock_auth_resp.user = mock_user
        mock_auth_resp.session = mock_session

        mock_client.auth.sign_in_with_password.return_value = mock_auth_resp
        mock_client.auth.admin.list_users.return_value = [mock_user]

        success, session_dict, msg = authenticate_admin(
            email="vickramabharathi@gmail.com",
            password="Admin#QRShield2026!",
            client=mock_client,
        )

        assert success is True
        assert session_dict is not None
        assert session_dict["role"] == "admin"
        assert session_dict["email"] == "vickramabharathi@gmail.com"
        assert verify_admin_authorization(session_dict) is True

    def test_invalid_credentials_rejected(self):
        """Invalid credentials must fail gracefully with access denied message."""
        mock_client = MagicMock()
        mock_client.auth.sign_in_with_password.side_effect = Exception("Invalid login credentials")
        mock_client.auth.admin.list_users.return_value = []

        success, session_dict, msg = authenticate_admin(
            email="vickramabharathi@gmail.com",
            password="WrongPassword999!",
            client=mock_client,
        )

        assert success is False
        assert session_dict is None
        assert "access denied" in msg.lower() or "invalid" in msg.lower()

    def test_unauthorized_user_cannot_gain_admin_access(self):
        """Standard authenticated user with role='user' is blocked from admin access."""
        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "standard-user-id"
        mock_user.email = "attacker@example.com"
        mock_user.user_metadata = {"role": "user"}
        mock_user.app_metadata = {"role": "user"}

        mock_auth_resp = MagicMock()
        mock_auth_resp.user = mock_user
        mock_auth_resp.session = MagicMock()

        mock_client.auth.sign_in_with_password.return_value = mock_auth_resp
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"role": "user"}
        ]

        success, session_dict, msg = authenticate_admin(
            email="attacker@example.com",
            password="SomePassword123!",
            client=mock_client,
        )

        assert success is False
        assert session_dict is None
        assert "privileges" in msg.lower() or "denied" in msg.lower()
        mock_client.auth.sign_out.assert_called_once()

    def test_logout_and_session_clearing(self):
        """Logging out revokes session validity."""
        mock_client = MagicMock()
        assert sign_out_admin(client=mock_client) is True
        mock_client.auth.sign_out.assert_called_once()

        # Session verification fails on None or empty dictionary
        assert verify_admin_authorization(None) is False
        assert verify_admin_authorization({}) is False
        assert verify_admin_authorization({"role": "user"}) is False

    def test_missing_credentials_or_unconfigured_supabase(self):
        """Empty or unconfigured client handles error safely without crashing."""
        success, session, msg = authenticate_admin("", "", client=None)
        assert success is False
        assert "email" in msg.lower()

        success2, session2, msg2 = authenticate_admin("admin@test.com", "", client=None)
        assert success2 is False
        assert "password" in msg2.lower()

    def test_public_user_cannot_access_global_scan_history(self):
        """Unauthenticated or public users are blocked from querying global audit history."""
        mock_client = MagicMock()
        scans = get_detailed_scan_history(limit=50, client=mock_client, is_admin=False)
        assert scans == []
        mock_client.table.assert_not_called()

    def test_export_csv_strictly_requires_admin(self):
        """Non-admin callers attempting to export scan records trigger PermissionError."""
        dummy_scans = [{"scan_id": "1", "extracted_url": "https://example.com"}]
        with pytest.raises(PermissionError):
            export_scan_history_to_csv(dummy_scans, is_admin=False)

        # Admin export succeeds
        csv_out = export_scan_history_to_csv(dummy_scans, is_admin=True)
        assert "scan_id" in csv_out
        assert "https://example.com" in csv_out

    def test_restricted_views_defined_and_protected(self):
        """The 4 mandatory restricted sections are defined and blocked for non-admins."""
        from app import ADMIN_ONLY_VIEWS, enforce_authorization_guard

        assert "threat_analysis" in ADMIN_ONLY_VIEWS
        assert "scan_history" in ADMIN_ONLY_VIEWS
        assert "threat_analytics" in ADMIN_ONLY_VIEWS
        assert "model_evaluation" in ADMIN_ONLY_VIEWS

        # In guest context, guard returns False
        with patch("app.is_admin_authenticated", return_value=False):
            assert enforce_authorization_guard("threat_analysis") is False
            assert enforce_authorization_guard("scan_history") is False
            assert enforce_authorization_guard("threat_analytics") is False
            assert enforce_authorization_guard("model_evaluation") is False

        # In admin context, guard returns True
        with patch("app.is_admin_authenticated", return_value=True):
            assert enforce_authorization_guard("threat_analysis") is True
            assert enforce_authorization_guard("scan_history") is True
            assert enforce_authorization_guard("threat_analytics") is True
            assert enforce_authorization_guard("model_evaluation") is True

    def test_preservation_of_qr_scanning_and_blocking_rules(self):
        """Malicious URLs are blocked unconditionally without open links."""
        engine = get_prevention_engine()
        mal_url = "http://192.168.1.1/update-credential-bank-login.php"
        analysis = analyze_url(mal_url)
        nav = engine.evaluate_navigation_decision(
            raw_payload=mal_url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
        )
        assert nav["is_permitted"] is False
        assert nav["status_category"] == "BLOCKED"
        assert nav["open_url"] is None
        assert nav["button_label"] is None

    def test_change_admin_password_success_mocked(self):
        """Administrator password change succeeds via Supabase Auth admin API."""
        mock_client = MagicMock()
        mock_auth_resp = MagicMock()
        mock_auth_resp.user = MagicMock()
        mock_client.auth.sign_in_with_password.return_value = mock_auth_resp
        mock_client.auth.admin.update_user_by_id.return_value = MagicMock()

        ok, msg = change_admin_password(
            new_password="NewSecure#Password2026!",
            current_password="Admin#QRShield2026!",
            confirm_password="NewSecure#Password2026!",
            user_id="mock-user-1234",
            client=mock_client,
        )
        assert ok is True
        assert "success" in msg.lower()
        mock_client.auth.admin.update_user_by_id.assert_called_once()

    def test_change_admin_password_mismatch(self):
        """Mismatched new password and confirmation are rejected."""
        ok, msg = change_admin_password(
            new_password="NewSecure#Password2026!",
            confirm_password="Different#Password2026!",
            client=MagicMock(),
        )
        assert ok is False
        assert "match" in msg.lower()

    def test_change_admin_password_identical_rejected(self):
        """Setting new password identical to current password is rejected."""
        ok, msg = change_admin_password(
            new_password="Admin#QRShield2026!",
            current_password="Admin#QRShield2026!",
            confirm_password="Admin#QRShield2026!",
            client=MagicMock(),
        )
        assert ok is False
        assert "different" in msg.lower()

    def test_change_admin_password_invalid_current(self):
        """Failing current password verification rejects change."""
        mock_client = MagicMock()
        mock_client.auth.sign_in_with_password.side_effect = Exception("Invalid login credentials")

        ok, msg = change_admin_password(
            new_password="NewSecure#Password2026!",
            current_password="WrongCurrentPassword123!",
            confirm_password="NewSecure#Password2026!",
            user_id="mock-user-1234",
            client=mock_client,
        )
        assert ok is False
        assert "incorrect" in msg.lower() or "verification failed" in msg.lower()

    def test_change_admin_password_weak_rejected(self):
        """Weak new passwords are rejected by complexity validation."""
        ok, msg = change_admin_password(
            new_password="weak",
            confirm_password="weak",
            client=MagicMock(),
        )
        assert ok is False
        assert "8 characters" in msg.lower()

    def test_change_admin_password_call_with_app_signature(self):
        """Test exact keyword signature called in app.py settings screen."""
        mock_client = MagicMock()
        mock_auth_resp = MagicMock()
        mock_auth_resp.user = MagicMock()
        mock_client.auth.sign_in_with_password.return_value = mock_auth_resp
        mock_client.auth.admin.update_user_by_id.return_value = MagicMock()

        # Exactly as invoked in app.py render_settings_screen
        ok, msg = change_admin_password(
            new_password="Fresh#SecurePassword2026!",
            current_password="Admin#QRShield2026!",
            confirm_password="Fresh#SecurePassword2026!",
            user_id="user-uuid-9999",
            client=mock_client,
        )
        assert ok is True
        assert "successfully" in msg.lower()

    def test_change_admin_password_network_or_unexpected_failure(self):
        """Unexpected exception in Supabase Auth returns clean error without leaking raw tracebacks."""
        mock_client = MagicMock()
        mock_client.auth.sign_in_with_password.return_value = MagicMock()
        mock_client.auth.admin.update_user_by_id.side_effect = RuntimeError("Supabase Auth API unavailable")

        ok, msg = change_admin_password(
            new_password="Fresh#SecurePassword2026!",
            current_password="Admin#QRShield2026!",
            confirm_password="Fresh#SecurePassword2026!",
            user_id="user-uuid-9999",
            client=mock_client,
        )
        assert ok is False
        assert "failed to update password" in msg.lower() or "unable to update password" in msg.lower()
        # Verify no python syntax traceback in user message
        assert "Traceback (most recent call last)" not in msg

    def test_get_user_profile_success(self):
        """Test retrieving structured profile from auth metadata and public table."""
        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.id = "user-1234"
        mock_user.email = "admin@qrshield.io"
        mock_user.created_at = "2026-01-01T00:00:00Z"
        mock_user.last_sign_in_at = "2026-10-18T10:00:00Z"
        mock_user.user_metadata = {"full_name": "Cyber SecOps", "role": "admin"}
        mock_client.auth.admin.get_user_by_id.return_value = MagicMock(user=mock_user)

        # Mock table return
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "id": "user-1234",
            "full_name": "Cyber SecOps",
            "role": "admin",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-10-18T10:00:00Z",
        }
        mock_client.table.return_value = mock_table

        profile = get_user_profile("user-1234", client=mock_client)
        assert profile is not None
        assert profile["id"] == "user-1234"
        assert profile["full_name"] == "Cyber SecOps"
        assert profile["role"] == "admin"
        assert profile["email"] == "admin@qrshield.io"
        assert profile["account_status"] == "Active"

    def test_update_user_profile_validation_and_execution(self):
        """Test updating profile full name with validation."""
        mock_client = MagicMock()
        # Empty user id rejected
        ok, msg = update_user_profile("", "New Name", client=mock_client)
        assert ok is False

        # Empty name rejected
        ok, msg = update_user_profile("user-1234", "", client=mock_client)
        assert ok is False

        # Valid update
        mock_client.auth.admin.update_user_by_id.return_value = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = MagicMock(data=[{"id": "user-1234"}])
        ok, msg = update_user_profile("user-1234", "Security Director", client=mock_client)
        assert ok is True
        assert "successfully" in msg.lower()

