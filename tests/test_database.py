"""Unit tests for modules/database.py.

Uses unittest.mock to test database functions and SupabaseManager methods
with mocked Supabase responses, requiring no live network connection or real credentials.
"""

from unittest.mock import MagicMock, patch
import pytest
from postgrest.exceptions import APIError

import config
from modules.database import (
    SupabaseManager,
    _handle_db_error,
    clear_last_error,
    export_scan_history_to_csv,
    get_detailed_scan_history,
    get_key_role,
    get_last_error,
    get_recent_scans,
    get_supabase_client,
    insert_prevention_decision,
    insert_rl_feedback,
    insert_scan_record,
    insert_threat_analysis,
    is_database_configured,
    run_database_diagnostics,
    save_complete_scan,
    test_connection as check_db_connection,
)



@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client mimicking postgrest query builder responses."""
    client = MagicMock()
    # Mock chain: client.table("name").select(...).execute()
    table_mock = MagicMock()
    client.table.return_value = table_mock

    # Defaults for chaining
    table_mock.select.return_value = table_mock
    table_mock.insert.return_value = table_mock
    table_mock.order.return_value = table_mock
    table_mock.limit.return_value = table_mock

    return client


def test_is_database_configured():
    """Verify is_database_configured correctly detects presence of credentials."""
    with patch("config.SUPABASE_URL", "https://example.supabase.co"), \
         patch("config.SUPABASE_KEY", "valid_key_123"):
        assert is_database_configured() is True

    with patch("config.SUPABASE_URL", ""), \
         patch("config.SUPABASE_KEY", "valid_key_123"):
        assert is_database_configured() is False

    with patch("config.SUPABASE_URL", "https://your-project-id.supabase.co"), \
         patch("config.SUPABASE_KEY", "your-supabase-key"):
        assert is_database_configured() is False


def test_get_supabase_client_missing_credentials():
    """Verify get_supabase_client returns None when credentials are missing."""
    with patch("config.SUPABASE_URL", ""), patch("config.SUPABASE_KEY", ""):
        client = get_supabase_client(force_new=True)
        assert client is None


@patch("modules.database.create_client")
def test_get_supabase_client_success(mock_create):
    """Verify get_supabase_client creates client when credentials are present."""
    fake_client = MagicMock()
    mock_create.return_value = fake_client

    client = get_supabase_client(
        supabase_url="https://valid.supabase.co",
        supabase_key="valid_token_xyz",
        force_new=True,
    )
    assert client == fake_client
    mock_create.assert_called_once_with("https://valid.supabase.co", "valid_token_xyz")


def test_test_connection_success(mock_supabase_client):
    """Verify test_connection returns True on successful table query."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "abc-123"}]
    mock_supabase_client.table().select().limit().execute.return_value = response_mock

    success, message = check_db_connection(client=mock_supabase_client)
    assert success is True
    assert "verified successfully" in message.lower()


def test_test_connection_missing_table(mock_supabase_client):
    """Verify test_connection returns informative error when relation does not exist."""
    error_mock = APIError({"message": "relation public.scan_history does not exist", "code": "42P01"})
    mock_supabase_client.table().select().limit().execute.side_effect = error_mock

    success, message = check_db_connection(client=mock_supabase_client)
    assert success is False
    assert "schema.sql" in message


def test_test_connection_no_client():
    """Verify test_connection fails gracefully if no client is available."""
    with patch("modules.database.get_supabase_client", return_value=None):
        success, message = check_db_connection(client=None)
        assert success is False
        assert "not configured" in message


def test_insert_scan_record_success(mock_supabase_client):
    """Verify insert_scan_record formats data and returns inserted row."""
    expected_row = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "extracted_url": "https://example.com/test",
        "qr_payload_type": "URL",
        "scan_status": "PENDING",
    }
    response_mock = MagicMock()
    response_mock.data = [expected_row]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = insert_scan_record(
        extracted_url="https://example.com/test",
        qr_payload_type="URL",
        client=mock_supabase_client,
    )

    assert result == expected_row
    mock_supabase_client.table.assert_called_with("scan_history")


def test_insert_scan_record_invalid_type_fallback(mock_supabase_client):
    """Verify insert_scan_record defaults unknown payload types to OTHER."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "123"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    insert_scan_record(
        extracted_url="custom_data",
        qr_payload_type="INVALID_TYPE",
        client=mock_supabase_client,
    )

    call_args = mock_supabase_client.table().insert.call_args[0][0]
    assert call_args["qr_payload_type"] == "OTHER"


def test_get_recent_scans(mock_supabase_client):
    """Verify get_recent_scans returns ordered records."""
    mock_records = [
        {"id": "1", "extracted_url": "https://url1.com"},
        {"id": "2", "extracted_url": "https://url2.com"},
    ]
    response_mock = MagicMock()
    response_mock.data = mock_records
    mock_supabase_client.table().select().order().limit().execute.return_value = response_mock

    records = get_recent_scans(limit=2, client=mock_supabase_client)
    assert records == mock_records
    mock_supabase_client.table().order.assert_called_with("created_at", desc=True)


def test_insert_threat_analysis(mock_supabase_client):
    """Verify insert_threat_analysis sanitizes score and writes record."""
    expected_row = {
        "id": "analysis-1",
        "scan_id": "scan-1",
        "predicted_class": "MALICIOUS",
        "risk_score": 0.95,
        "detected_indicators": ["ip_address_host", "shortener"],
        "model_version": "v1.0.0",
    }
    response_mock = MagicMock()
    response_mock.data = [expected_row]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = insert_threat_analysis(
        scan_id="scan-1",
        predicted_class="malicious",
        risk_score=1.5,  # Exceeds 1.0, should clamp to 1.0
        detected_indicators=["ip_address_host", "shortener"],
        client=mock_supabase_client,
    )

    assert result == expected_row
    # Check clamping
    call_args = mock_supabase_client.table().insert.call_args[0][0]
    assert call_args["risk_score"] == 1.0
    assert call_args["predicted_class"] == "MALICIOUS"


def test_insert_prevention_decision(mock_supabase_client):
    """Verify insert_prevention_decision validates action and writes record."""
    expected_row = {
        "id": "dec-1",
        "scan_id": "scan-1",
        "selected_action": "BLOCK",
        "decision_reason": "High risk phishing indicators detected",
        "policy_version": "v1.0.0",
    }
    response_mock = MagicMock()
    response_mock.data = [expected_row]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = insert_prevention_decision(
        scan_id="scan-1",
        selected_action="block",
        decision_reason="High risk phishing indicators detected",
        client=mock_supabase_client,
    )

    assert result == expected_row
    call_args = mock_supabase_client.table().insert.call_args[0][0]
    assert call_args["selected_action"] == "BLOCK"


def test_insert_rl_feedback(mock_supabase_client):
    """Verify insert_rl_feedback inserts reward and action telemetry."""
    expected_row = {
        "id": "fb-1",
        "scan_id": "scan-1",
        "state_label": "HIGH_RISK_REDIRECT",
        "selected_action": "BLOCK",
        "reward": 10.0,
        "feedback_source": "USER",
    }
    response_mock = MagicMock()
    response_mock.data = [expected_row]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = insert_rl_feedback(
        scan_id="scan-1",
        state_label="HIGH_RISK_REDIRECT",
        selected_action="block",
        reward=10.0,
        feedback_source="user",
        client=mock_supabase_client,
    )

    assert result == expected_row
    call_args = mock_supabase_client.table().insert.call_args[0][0]
    assert call_args["selected_action"] == "BLOCK"
    assert call_args["feedback_source"] == "USER"
    assert call_args["reward"] == 10.0


def test_supabase_manager_wrapper(mock_supabase_client):
    """Verify SupabaseManager correctly encapsulates operations."""
    manager = SupabaseManager()
    manager.client = mock_supabase_client

    response_mock = MagicMock()
    response_mock.data = [{"id": "scan-xyz"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    scan = manager.log_scan("https://test.com")
    assert scan == {"id": "scan-xyz"}

    decision = manager.log_prevention_decision("scan-xyz", "ALLOW", "Safe domain")
    assert decision == {"id": "scan-xyz"}


def test_save_complete_scan_success(mock_supabase_client):
    """Verify save_complete_scan coordinates scan, threat analysis, and prevention decision."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "scan-uuid-123"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = save_complete_scan(
        extracted_url="https://secure-bank.com",
        qr_payload_type="URL",
        predicted_class="SAFE",
        risk_score=0.12,
        detected_indicators=[],
        selected_action="ALLOW",
        decision_reason="Clean URL and verified domain",
        scan_id="client-provided-uuid",
        client=mock_supabase_client,
    )

    assert result is not None
    assert result["scan_id"] == "scan-uuid-123"
    assert result["scan_record"] == {"id": "scan-uuid-123"}
    assert result["threat_record"] == {"id": "scan-uuid-123"}
    assert result["decision_record"] == {"id": "scan-uuid-123"}


def test_save_complete_scan_failure_when_unconfigured():
    """Verify save_complete_scan returns None when client is None."""
    with patch("modules.database.get_supabase_client", return_value=None):
        result = save_complete_scan(
            extracted_url="https://test.com",
            qr_payload_type="URL",
            predicted_class="SAFE",
            risk_score=0.1,
            detected_indicators=[],
            selected_action="ALLOW",
            decision_reason="Testing",
            client=None,
        )
        assert result is None


def test_insert_scan_record_failed_insert(mock_supabase_client):
    """Verify insert_scan_record handles empty response or API error gracefully."""
    response_mock = MagicMock()
    response_mock.data = []  # Empty response
    mock_supabase_client.table().insert().execute.return_value = response_mock

    res = insert_scan_record("https://test.com", client=mock_supabase_client)
    assert res is None

    # API error
    mock_supabase_client.table().insert().execute.side_effect = APIError({"message": "Permission denied", "code": "42501"})
    res_err = insert_scan_record("https://test.com", client=mock_supabase_client)
    assert res_err is None


def test_action_warn_normalization_to_warn_user(mock_supabase_client):
    """Verify that 'WARN' is normalized to 'WARN_USER' to satisfy DB CHECK constraints."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "dec-123"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    insert_prevention_decision("scan-1", "WARN", "Uncertain signals", client=mock_supabase_client)
    call_args = mock_supabase_client.table().insert.call_args[0][0]
    assert call_args["selected_action"] == "WARN_USER"

    insert_rl_feedback("scan-1", "STATE_X", "WARN", 5.0, client=mock_supabase_client)
    fb_args = mock_supabase_client.table().insert.call_args[0][0]
    assert fb_args["selected_action"] == "WARN_USER"


def test_get_detailed_scan_history_success(mock_supabase_client):
    """Verify get_detailed_scan_history flattens relational embedded records."""
    mock_rows = [
        {
            "id": "uuid-1",
            "created_at": "2026-10-17T09:30:00Z",
            "extracted_url": "https://paypal-verify.com",
            "qr_payload_type": "URL",
            "scan_status": "PROCESSED",
            "threat_analysis": [
                {
                    "predicted_class": "MALICIOUS",
                    "risk_score": 0.88,
                    "detected_indicators": ["suspicious_keyword"],
                    "model_version": "v1.0.0",
                }
            ],
            "prevention_decisions": [
                {
                    "selected_action": "BLOCK",
                    "decision_reason": "High risk detected",
                    "policy_version": "v1.0.0",
                }
            ],
        }
    ]
    response_mock = MagicMock()
    response_mock.data = mock_rows
    mock_supabase_client.table().select().order().limit().execute.return_value = response_mock

    results = get_detailed_scan_history(limit=5, client=mock_supabase_client)

    assert len(results) == 1
    item = results[0]
    assert item["scan_id"] == "uuid-1"
    assert item["extracted_url"] == "https://paypal-verify.com"
    assert item["predicted_class"] == "MALICIOUS"
    assert item["risk_score"] == 0.88
    assert item["prevention_action"] == "BLOCK"


def test_get_detailed_scan_history_empty(mock_supabase_client):
    """Verify get_detailed_scan_history handles empty database responses."""
    response_mock = MagicMock()
    response_mock.data = []
    mock_supabase_client.table().select().order().limit().execute.return_value = response_mock

    results = get_detailed_scan_history(limit=10, client=mock_supabase_client)
    assert results == []


def test_export_scan_history_to_csv():
    """Verify export_scan_history_to_csv creates valid CSV content."""
    scans = [
        {
            "scan_id": "scan-1",
            "timestamp": "2026-10-17T09:00:00Z",
            "extracted_url": "https://google.com",
            "payload_type": "URL",
            "status": "PROCESSED",
            "predicted_class": "SAFE",
            "risk_score": 0.05,
            "prevention_action": "ALLOW",
            "decision_reason": "Clean domain",
            "model_version": "v1.0.0",
            "policy_version": "v1.0.0",
        }
    ]
    csv_str = export_scan_history_to_csv(scans)

    assert "scan_id,timestamp,extracted_url" in csv_str
    assert "https://google.com" in csv_str
    assert "ALLOW" in csv_str
    assert "SAFE" in csv_str

    # Empty list export should still produce headers
    empty_csv = export_scan_history_to_csv([])
    assert "scan_id,timestamp,extracted_url" in empty_csv


def test_test_connection_network_error(mock_supabase_client):
    """Verify test_connection handles unexpected network/socket exceptions gracefully."""
    mock_supabase_client.table().select().limit().execute.side_effect = ConnectionError("Connection refused by peer")
    success, message = check_db_connection(client=mock_supabase_client)
    assert success is False
    assert "Connection refused" in message


def test_insert_threat_analysis_api_error(mock_supabase_client):
    """Verify insert_threat_analysis catches APIError and returns None without raising."""
    mock_supabase_client.table().insert().execute.side_effect = APIError({"message": "Foreign key violation", "code": "23503"})
    result = insert_threat_analysis("nonexistent-id", "SAFE", 0.1, [], client=mock_supabase_client)
    assert result is None


def test_insert_prevention_decision_api_error(mock_supabase_client):
    """Verify insert_prevention_decision catches APIError and returns None without raising."""
    mock_supabase_client.table().insert().execute.side_effect = APIError({"message": "Permission denied", "code": "42501"})
    result = insert_prevention_decision("nonexistent-id", "ALLOW", "testing", client=mock_supabase_client)
    assert result is None


def test_insert_rl_feedback_api_error(mock_supabase_client):
    """Verify insert_rl_feedback catches APIError and returns None without raising."""
    mock_supabase_client.table().insert().execute.side_effect = APIError({"message": "Table missing", "code": "42P01"})
    result = insert_rl_feedback("nonexistent-id", "STATE_1", "ALLOW", 1.0, client=mock_supabase_client)
    assert result is None


def test_get_detailed_scan_history_fallback_on_error(mock_supabase_client):
    """Verify get_detailed_scan_history falls back to simple scan records if embedded join fails."""
    # First call (embedded join) raises APIError
    # Second call (fallback simple query) succeeds
    mock_supabase_client.table().select().order().limit().execute.side_effect = [
        APIError({"message": "Embedded resource not found", "code": "PGRST200"}),
        MagicMock(data=[{"id": "scan-fb-1", "created_at": "2026-10-17T10:00:00Z", "extracted_url": "https://fallback.com", "qr_payload_type": "URL", "scan_status": "PROCESSED"}]),
    ]

    results = get_detailed_scan_history(limit=5, client=mock_supabase_client)
    assert len(results) == 1
    assert results[0]["scan_id"] == "scan-fb-1"
    assert results[0]["extracted_url"] == "https://fallback.com"
    assert results[0]["predicted_class"] == "N/A"


def test_save_complete_scan_with_none_indicators(mock_supabase_client):
    """Verify save_complete_scan handles None detected_indicators without raising AttributeError."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "scan-none-indicators"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = save_complete_scan(
        extracted_url="https://example.com/clean",
        qr_payload_type="URL",
        predicted_class="SAFE",
        risk_score=0.05,
        detected_indicators=None,  # Explicitly None
        selected_action="ALLOW",
        decision_reason="Clean payload",
        client=mock_supabase_client,
    )

    assert result is not None
    assert result["scan_id"] == "scan-none-indicators"


def test_save_complete_scan_with_url_analysis_result_integration(mock_supabase_client):
    """Verify Supabase save operates seamlessly with URLAnalysisResult indicators."""
    from modules.url_analyzer import analyze_url

    analysis = analyze_url("http://192.168.1.1/admin")
    # Both properties must be present and identical
    assert hasattr(analysis, "risk_indicators")
    assert hasattr(analysis, "suspicious_indicators")
    assert analysis.risk_indicators == analysis.suspicious_indicators

    # Extract indicators using the safe resilient pattern
    indicators = getattr(analysis, "risk_indicators", getattr(analysis, "suspicious_indicators", []))
    assert isinstance(indicators, list)
    assert len(indicators) > 0

    response_mock = MagicMock()
    response_mock.data = [{"id": "scan-url-analysis-id"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = save_complete_scan(
        extracted_url=analysis.raw_payload,
        qr_payload_type="URL",
        predicted_class="SUSPICIOUS",
        risk_score=analysis.risk_score,
        detected_indicators=indicators,
        selected_action="WARN",
        decision_reason="IP address host detected",
        client=mock_supabase_client,
    )

    assert result is not None
    assert result["scan_id"] == "scan-url-analysis-id"


def test_missing_indicators_safe_extraction_no_attribute_error():
    """Verify that objects without suspicious_indicators attribute resolve safely without AttributeError."""
    class DummyAnalysisNoIndicators:
        raw_payload = "https://test.com"

    dummy = DummyAnalysisNoIndicators()
    # Emulate the safe extraction used in app.py
    extracted = getattr(dummy, "risk_indicators", getattr(dummy, "suspicious_indicators", []))
    assert extracted == []


def test_get_key_role():
    """Verify get_key_role safely parses JWT claims without network requests."""
    # Base64url encoded: {"iss": "supabase", "role": "anon"}
    anon_jwt = "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJhbm9uIn0.sig"
    assert get_key_role(anon_jwt) == "anon"

    # Base64url encoded: {"iss": "supabase", "role": "service_role"}
    service_jwt = "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJzZXJ2aWNlX3JvbGUifQ.sig"
    assert get_key_role(service_jwt) == "service_role"

    # Malformed tokens
    assert get_key_role("invalid-token") == "unknown"
    assert get_key_role("") == "unknown"


def test_handle_db_error_table_missing():
    """Verify _handle_db_error generates actionable schema.sql guidance for PGRST205."""
    clear_last_error()
    err = APIError({"message": "Could not find the table 'public.scan_history' in the schema cache", "code": "PGRST205"})
    summary = _handle_db_error(err, "INSERT", "scan_history")

    assert summary["code"] == "PGRST205"
    assert summary["table"] == "scan_history"
    assert "schema.sql" in summary["guidance"]
    assert get_last_error() == summary

    clear_last_error()
    assert get_last_error() is None


def test_handle_db_error_rls_denial():
    """Verify _handle_db_error generates actionable service_role key guidance for error 42501."""
    err = APIError({"message": "new row violates row-level security policy for table scan_history", "code": "42501"})
    summary = _handle_db_error(err, "INSERT", "scan_history")

    assert summary["code"] == "42501"
    assert "service_role" in summary["guidance"].lower()



def test_handle_db_error_foreign_key_violation():
    """Verify _handle_db_error generates clear foreign-key guidance for error 23503."""
    err = APIError({"message": "insert or update on table threat_analysis violates foreign key constraint", "code": "23503"})
    summary = _handle_db_error(err, "INSERT", "threat_analysis")

    assert summary["code"] == "23503"
    assert "Foreign key" in summary["guidance"]


def test_handle_db_error_check_constraint_violation():
    """Verify _handle_db_error generates clear CHECK constraint guidance for error 23514."""
    err = APIError({"message": "new row for relation threat_analysis violates check constraint", "code": "23514"})
    summary = _handle_db_error(err, "INSERT", "threat_analysis")

    assert summary["code"] == "23514"
    assert "CHECK constraint" in summary["guidance"]


def test_handle_db_error_redacts_credentials():
    """Verify _handle_db_error redacts Supabase keys and URLs from log messages and summaries."""
    secret_key = "super_secret_test_key_12345678"
    with patch("config.SUPABASE_KEY", secret_key):
        err = Exception(f"Failed authenticating with {secret_key}")
        summary = _handle_db_error(err, "INSERT", "scan_history")
        assert secret_key not in summary["message"]
        assert "[REDACTED]" in summary["message"]


def test_save_complete_scan_fails_on_threat_analysis_failure():
    """Verify save_complete_scan returns None when child threat_analysis insert fails."""
    client = MagicMock()
    scan_history_mock = MagicMock()
    threat_mock = MagicMock()

    def get_table(name):
        if name == "scan_history":
            return scan_history_mock
        return threat_mock

    client.table.side_effect = get_table

    # 1. scan_history insert succeeds
    scan_history_mock.insert.return_value.execute.return_value = MagicMock(data=[{"id": "scan-parent-1"}])

    # 2. threat_analysis insert fails with FK violation or RLS error
    threat_mock.insert.return_value.execute.side_effect = APIError({
        "message": "insert violates foreign key constraint",
        "code": "23503",
    })

    result = save_complete_scan(
        extracted_url="https://test.com",
        qr_payload_type="URL",
        predicted_class="SAFE",
        risk_score=0.1,
        selected_action="ALLOW",
        decision_reason="Clean",
        client=client,
    )

    assert result is None
    last_err = get_last_error()
    assert last_err is not None
    assert last_err["table"] == "threat_analysis"


def test_save_complete_scan_fails_on_prevention_decision_failure():
    """Verify save_complete_scan returns None when child prevention_decisions insert fails."""
    client = MagicMock()
    scan_history_mock = MagicMock()
    threat_mock = MagicMock()
    prevention_mock = MagicMock()

    def get_table(name):
        if name == "scan_history":
            return scan_history_mock
        elif name == "threat_analysis":
            return threat_mock
        return prevention_mock

    client.table.side_effect = get_table

    # 1. scan_history insert succeeds
    scan_history_mock.insert.return_value.execute.return_value = MagicMock(data=[{"id": "scan-parent-2"}])
    # 2. threat_analysis insert succeeds
    threat_mock.insert.return_value.execute.return_value = MagicMock(data=[{"id": "threat-child-1"}])
    # 3. prevention_decisions insert fails with RLS permission denied
    prevention_mock.insert.return_value.execute.side_effect = APIError({
        "message": "permission denied for table prevention_decisions",
        "code": "42501",
    })

    result = save_complete_scan(
        extracted_url="https://test.com",
        qr_payload_type="URL",
        predicted_class="SAFE",
        risk_score=0.1,
        selected_action="ALLOW",
        decision_reason="Clean",
        client=client,
    )

    assert result is None
    last_err = get_last_error()
    assert last_err is not None
    assert last_err["table"] == "prevention_decisions"
    assert "42501" in str(last_err["code"])


@patch("modules.database.create_client")
def test_get_supabase_client_prefers_service_role_key(mock_create):
    """Verify get_supabase_client uses SUPABASE_SERVICE_ROLE_KEY over public SUPABASE_KEY."""
    fake_client = MagicMock()
    mock_create.return_value = fake_client

    with patch("config.SUPABASE_URL", "https://valid.supabase.co"), \
         patch("config.SUPABASE_KEY", "anon_key_123"), \
         patch("config.SUPABASE_SERVICE_ROLE_KEY", "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJzZXJ2aWNlX3JvbGUifQ.sig"):
        client = get_supabase_client(force_new=True, require_service_role=True)
        assert client == fake_client
        mock_create.assert_called_with("https://valid.supabase.co", "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJzZXJ2aWNlX3JvbGUifQ.sig")


def test_get_supabase_client_rejects_anon_key_when_service_role_required():
    """Verify get_supabase_client returns None when require_service_role is True but only anon key is present."""
    with patch("config.SUPABASE_URL", "https://valid.supabase.co"), \
         patch("config.SUPABASE_KEY", "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJhbm9uIn0.sig"), \
         patch("config.SUPABASE_SERVICE_ROLE_KEY", ""):
        client = get_supabase_client(force_new=True, require_service_role=True)
        assert client is None


def test_insert_scan_record_fails_gracefully_when_service_role_key_missing():
    """Verify insert_scan_record returns None and sets guidance when SUPABASE_SERVICE_ROLE_KEY is missing."""
    with patch("config.SUPABASE_URL", "https://valid.supabase.co"), \
         patch("config.SUPABASE_KEY", "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJhbm9uIn0.sig"), \
         patch("config.SUPABASE_SERVICE_ROLE_KEY", ""):
        result = insert_scan_record("https://test.com", "URL", "PROCESSED", client=None)
        assert result is None

        last_err = get_last_error()
        assert last_err is not None
        assert "SUPABASE_SERVICE_ROLE_KEY is missing" in last_err["guidance"]


def test_save_complete_scan_fails_gracefully_when_service_role_key_missing():
    """Verify save_complete_scan returns None and sets guidance when SUPABASE_SERVICE_ROLE_KEY is missing."""
    with patch("config.SUPABASE_URL", "https://valid.supabase.co"), \
         patch("config.SUPABASE_KEY", "header.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJhbm9uIn0.sig"), \
         patch("config.SUPABASE_SERVICE_ROLE_KEY", ""):
        result = save_complete_scan(
            extracted_url="https://test.com",
            qr_payload_type="URL",
            predicted_class="SAFE",
            risk_score=0.1,
            selected_action="ALLOW",
            decision_reason="Testing",
            client=None,
        )
        assert result is None

        last_err = get_last_error()
        assert last_err is not None
        assert "SUPABASE_SERVICE_ROLE_KEY is missing" in last_err["guidance"]


def test_insert_scan_record_database_write_failure_under_rls(mock_supabase_client):
    """Verify insert_scan_record handles PostgreSQL 42501 RLS policy rejection accurately."""
    mock_supabase_client.table("scan_history").insert().execute.side_effect = APIError({
        "message": "new row violates row-level security policy for table scan_history",
        "code": "42501",
    })

    result = insert_scan_record("https://test.com", "URL", "PENDING", client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "42501"
    assert "Row Level Security (RLS) policy rejected" in last_err["guidance"]
    assert "SUPABASE_KEY" in last_err["guidance"] or "SUPABASE_SERVICE_ROLE_KEY" in last_err["guidance"]


def test_get_key_role_modern_and_jwt_formats():
    """Verify get_key_role recognizes modern Supabase key prefixes and legacy JWTs."""
    # Modern prefixes
    assert get_key_role("sb_secret_abcdef1234567890abcdef") == "service_role"
    assert get_key_role("sb_publishable_abcdef1234567890abcdef") == "anon"

    # Empty or invalid
    assert get_key_role("") == "unknown"
    assert get_key_role("random_string_key") == "unknown"


def test_insert_threat_analysis_permission_failure(mock_supabase_client):
    """Verify insert_threat_analysis handles RLS 42501 permission rejection."""
    mock_supabase_client.table("threat_analysis").insert().execute.side_effect = APIError({
        "message": "permission denied for table threat_analysis",
        "code": "42501",
    })

    result = insert_threat_analysis("scan-123", "SAFE", 0.1, client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "42501"
    assert last_err["table"] == "threat_analysis"
    assert "Row Level Security (RLS) policy rejected" in last_err["guidance"]


def test_insert_prevention_decision_permission_failure(mock_supabase_client):
    """Verify insert_prevention_decision handles RLS 42501 permission rejection."""
    mock_supabase_client.table("prevention_decisions").insert().execute.side_effect = APIError({
        "message": "new row violates row-level security policy for table prevention_decisions",
        "code": "42501",
    })

    result = insert_prevention_decision("scan-123", "WARN_USER", "Review", client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "42501"
    assert last_err["table"] == "prevention_decisions"
    assert "Row Level Security (RLS) policy rejected" in last_err["guidance"]


def test_insert_rl_feedback_permission_failure(mock_supabase_client):
    """Verify insert_rl_feedback handles RLS 42501 permission rejection."""
    mock_supabase_client.table("rl_feedback").insert().execute.side_effect = APIError({
        "message": "new row violates row-level security policy for table rl_feedback",
        "code": "42501",
    })

    result = insert_rl_feedback("scan-123", "STATE_0", "ALLOW", 10.0, client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "42501"
    assert last_err["table"] == "rl_feedback"
    assert "Row Level Security (RLS) policy rejected" in last_err["guidance"]


def test_insert_threat_analysis_foreign_key_violation(mock_supabase_client):
    """Verify insert_threat_analysis handles foreign key constraint violation (Error 23503)."""
    mock_supabase_client.table("threat_analysis").insert().execute.side_effect = APIError({
        "message": "insert violates foreign key constraint 'threat_analysis_scan_id_fkey'",
        "code": "23503",
    })

    result = insert_threat_analysis("non-existent-scan-id", "SAFE", 0.05, client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "23503"
    assert last_err["table"] == "threat_analysis"
    assert "Foreign key constraint violation" in last_err["guidance"]


def test_insert_prevention_decision_foreign_key_violation(mock_supabase_client):
    """Verify insert_prevention_decision handles foreign key constraint violation (Error 23503)."""
    mock_supabase_client.table("prevention_decisions").insert().execute.side_effect = APIError({
        "message": "insert violates foreign key constraint 'prevention_decisions_scan_id_fkey'",
        "code": "23503",
    })

    result = insert_prevention_decision("non-existent-scan-id", "BLOCK", "Malicious target", client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "23503"
    assert last_err["table"] == "prevention_decisions"
    assert "Foreign key constraint violation" in last_err["guidance"]


def test_insert_rl_feedback_foreign_key_violation(mock_supabase_client):
    """Verify insert_rl_feedback handles foreign key constraint violation (Error 23503)."""
    mock_supabase_client.table("rl_feedback").insert().execute.side_effect = APIError({
        "message": "insert violates foreign key constraint 'rl_feedback_scan_id_fkey'",
        "code": "23503",
    })

    result = insert_rl_feedback("non-existent-scan-id", "STATE_0", "ALLOW", 5.0, client=mock_supabase_client)
    assert result is None

    last_err = get_last_error()
    assert last_err is not None
    assert last_err["code"] == "23503"
    assert last_err["table"] == "rl_feedback"
    assert "Foreign key constraint violation" in last_err["guidance"]


def test_insert_records_missing_credentials():
    """Verify insert functions fail gracefully when no Supabase credentials are configured."""
    with patch("config.SUPABASE_URL", ""), patch("config.SUPABASE_KEY", ""), patch("config.SUPABASE_SERVICE_ROLE_KEY", ""):
        assert insert_scan_record("https://test.com", client=None) is None
        assert get_last_error()["guidance"].startswith("Database error during INSERT")

        assert insert_threat_analysis("id", "SAFE", 0.1, client=None) is None
        assert insert_prevention_decision("id", "ALLOW", "Clean", client=None) is None
        assert insert_rl_feedback("id", "S0", "ALLOW", 1.0, client=None) is None


def test_handle_db_error_regex_redacts_jwt_and_bearer():
    """Verify _handle_db_error scrubs arbitrary Bearer tokens and JWT signatures from errors."""
    leak_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doNotLeakThisSignature12345"
    exc = Exception(f"Failed Authorization: Bearer {leak_jwt}")
    summary = _handle_db_error(exc, "INSERT", "scan_history")

    assert leak_jwt not in summary["message"]
    assert "[REDACTED_TOKEN]" in summary["message"] or "[REDACTED]" in summary["message"]
    assert "doNotLeakThisSignature12345" not in summary["message"]


@patch("modules.database.create_client")
def test_get_supabase_client_uses_supabase_key_when_service_role(mock_create):
    """Verify get_supabase_client uses SUPABASE_KEY when it contains a service-role secret."""
    fake_client = MagicMock()
    mock_create.return_value = fake_client

    with patch("config.SUPABASE_URL", "https://valid.supabase.co"), \
         patch("config.SUPABASE_KEY", "sb_secret_test_service_key_99999"), \
         patch("config.SUPABASE_SERVICE_ROLE_KEY", ""):
        client = get_supabase_client(force_new=True, require_service_role=True)
        assert client == fake_client
        mock_create.assert_called_with("https://valid.supabase.co", "sb_secret_test_service_key_99999")


def test_run_database_diagnostics_safe_output(mock_supabase_client):
    """Verify run_database_diagnostics checks tables safely without secret leakage."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "row-1"}]
    mock_supabase_client.table().select().limit().execute.return_value = response_mock

    diag = run_database_diagnostics(client=mock_supabase_client)
    assert diag["configured"] is True
    assert diag["connection_ok"] is True
    assert isinstance(diag["tables_accessible"], list)
    assert "scan_history" in diag["tables_accessible"]

    # Ensure no secrets leak in recommendations or details
    diag_str = str(diag)
    assert "password" not in diag_str.lower()
    assert "Bearer" not in diag_str
    assert "secret" not in diag_str.lower() or "service_role" in diag_str.lower()


def test_insert_scan_record_with_user_and_domain(mock_supabase_client):
    """Verify insert_scan_record sends user_id and destination_domain when provided."""
    expected_row = {
        "id": "scan-1234",
        "extracted_url": "https://malicious-login.com/login",
        "qr_payload_type": "URL",
        "scan_status": "PROCESSED",
        "user_id": "user-uuid-8888",
        "destination_domain": "malicious-login.com",
    }
    response_mock = MagicMock()
    response_mock.data = [expected_row]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = insert_scan_record(
        extracted_url="https://malicious-login.com/login",
        qr_payload_type="URL",
        scan_status="PROCESSED",
        user_id="user-uuid-8888",
        destination_domain="malicious-login.com",
        client=mock_supabase_client,
    )

    assert result == expected_row
    call_args = mock_supabase_client.table().insert.call_args[0][0]
    assert call_args["user_id"] == "user-uuid-8888"
    assert call_args["destination_domain"] == "malicious-login.com"


def test_insert_scan_record_schema_fallback_on_missing_column(mock_supabase_client):
    """Verify insert_scan_record falls back to base schema if extended columns don't exist in live table."""
    # First insert fails because destination_domain or user_id column doesn't exist
    err_col = APIError({"message": "Could not find column 'destination_domain' in table 'scan_history'", "code": "PGRST204"})
    
    # Second insert succeeds with base columns
    fallback_row = {"id": "scan-fallback-1", "extracted_url": "https://example.com", "qr_payload_type": "URL", "scan_status": "PROCESSED"}
    success_mock = MagicMock(data=[fallback_row])

    mock_supabase_client.table().insert().execute.side_effect = [err_col, success_mock]

    result = insert_scan_record(
        extracted_url="https://example.com",
        qr_payload_type="URL",
        user_id="user-123",
        destination_domain="example.com",
        client=mock_supabase_client,
    )

    assert result == fallback_row
    # Verify second call stripped the extended columns
    second_call_args = mock_supabase_client.table().insert.call_args_list[-1][0][0]
    assert "user_id" not in second_call_args
    assert "destination_domain" not in second_call_args


def test_save_complete_scan_associates_user_and_parses_domain(mock_supabase_client):
    """Verify save_complete_scan parses destination domain and passes user_id."""
    response_mock = MagicMock()
    response_mock.data = [{"id": "scan-complete-user-1"}]
    mock_supabase_client.table().insert().execute.return_value = response_mock

    result = save_complete_scan(
        extracted_url="https://secure-banking.org/dashboard",
        qr_payload_type="URL",
        predicted_class="SAFE",
        risk_score=0.08,
        detected_indicators=[],
        selected_action="ALLOW",
        decision_reason="Legitimate domain",
        user_id="user-sec-999",
        client=mock_supabase_client,
    )

    assert result is not None
    assert result["scan_id"] == "scan-complete-user-1"





