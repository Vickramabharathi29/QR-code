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
    export_scan_history_to_csv,
    get_detailed_scan_history,
    get_recent_scans,
    get_supabase_client,
    insert_prevention_decision,
    insert_rl_feedback,
    insert_scan_record,
    insert_threat_analysis,
    is_database_configured,
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

