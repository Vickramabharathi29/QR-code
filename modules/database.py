"""Database Module for QRShield.

Manages connections and transactions with Supabase PostgreSQL using the official
supabase Python client. Provides CRUD helper functions for:
- Scan history records
- URL threat analysis results
- Adaptive prevention decisions
- Reinforcement learning feedback

Features graceful fallback handling for missing credentials, network timeouts,
and PostgreSQL/PostgREST errors without crashing or falling back to SQLite.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from postgrest.exceptions import APIError
from supabase import Client, create_client

import config

logger = logging.getLogger("qrshield.database")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_client_instance: Optional[Client] = None


def is_database_configured() -> bool:
    """Check whether the Supabase database URL and API key are configured in environment.

    Returns:
        bool: True if non-empty, non-placeholder credentials exist.
    """
    return config.is_supabase_configured()


def get_supabase_client(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    force_new: bool = False,
) -> Optional[Client]:
    """Initialize and return the official Supabase Python client instance.

    Uses configured environment variables by default. Caches the client instance
    as a singleton unless force_new is True.

    Args:
        supabase_url: Optional custom URL override.
        supabase_key: Optional custom API key override.
        force_new: If True, bypasses the cached singleton client.

    Returns:
        Optional[Client]: The connected Supabase client, or None if credentials are missing/invalid.
    """
    global _client_instance

    if _client_instance is not None and not force_new and supabase_url is None and supabase_key is None:
        return _client_instance

    url = (supabase_url or config.SUPABASE_URL).strip()
    key = (supabase_key or config.SUPABASE_KEY).strip()

    if not url or not key:
        logger.warning("Supabase URL or API key is not configured. Database operations will be disabled.")
        return None

    if "your-project-id" in url or "your-supabase" in key:
        logger.warning("Default placeholder credentials detected in Supabase config. Please provide real credentials.")
        return None

    try:
        client = create_client(url, key)
        if not force_new and supabase_url is None and supabase_key is None:
            _client_instance = client
        return client
    except Exception as exc:
        logger.error(f"Failed to initialize Supabase client: {exc}", exc_info=True)
        return None


def test_connection(client: Optional[Client] = None) -> Tuple[bool, str]:
    """Verify connectivity with the Supabase PostgreSQL database.

    Args:
        client: Optional Supabase client instance. If None, uses get_supabase_client().

    Returns:
        Tuple[bool, str]: (Success boolean, Status or error message).
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        return False, "Supabase client is not configured or failed to initialize. Check .env settings."

    try:
        # Perform a lightweight query against the scan_history table
        response = active_client.table("scan_history").select("id").limit(1).execute()
        return True, "Supabase database connection verified successfully."
    except APIError as api_err:
        err_msg = str(api_err)
        if (
            ("relation" in err_msg.lower() and "does not exist" in err_msg.lower())
            or "could not find the table" in err_msg.lower()
            or "pgrst205" in err_msg.lower()
        ):
            return False, (
                "Connected to Supabase project, but tables do not exist yet. "
                "Please run supabase/schema.sql in the Supabase SQL Editor."
            )
        logger.error(f"Supabase APIError during connection test: {api_err}")
        return False, f"Supabase API Error: {api_err}"
    except Exception as exc:
        logger.error(f"Unexpected error during Supabase connection test: {exc}", exc_info=True)
        return False, f"Connection failed: {str(exc)}"


def insert_scan_record(
    extracted_url: Optional[str],
    qr_payload_type: str = "URL",
    scan_status: str = "PENDING",
    scan_id: Optional[str] = None,
    client: Optional[Client] = None,
) -> Optional[Dict[str, Any]]:
    """Insert a new QR scan event into the scan_history table.

    Args:
        extracted_url: The decoded URL or payload string from the QR code.
        qr_payload_type: Type of QR payload ('URL', 'TEXT', 'WIFI', 'VCARD', 'OTHER').
        scan_status: Status flag ('PENDING', 'PROCESSED', 'FAILED').
        scan_id: Optional client-specified UUID for idempotency.
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: The inserted row data containing the generated UUID, or None on failure.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot insert scan record: Supabase is not configured.")
        return None

    valid_payload_types = {"URL", "TEXT", "WIFI", "VCARD", "OTHER"}
    if qr_payload_type not in valid_payload_types:
        qr_payload_type = "OTHER"

    valid_statuses = {"PENDING", "PROCESSED", "FAILED"}
    if scan_status not in valid_statuses:
        scan_status = "PENDING"

    record: Dict[str, Any] = {
        "extracted_url": extracted_url,
        "qr_payload_type": qr_payload_type,
        "scan_status": scan_status,
    }
    if scan_id:
        record["id"] = scan_id

    try:
        response = active_client.table("scan_history").insert(record).execute()
        if response.data and len(response.data) > 0:
            inserted = response.data[0]
            logger.info(f"Recorded scan {inserted.get('id')} for payload: {extracted_url}")
            return inserted
        return None
    except APIError as api_err:
        logger.error(f"Database error inserting scan record: {api_err}")
        return None
    except Exception as exc:
        logger.error(f"Unexpected error inserting scan record: {exc}", exc_info=True)
        return None


def get_recent_scans(
    limit: int = 10,
    client: Optional[Client] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the most recent scan records from the scan_history table.

    Args:
        limit: Maximum number of records to retrieve (default: 10).
        client: Optional Supabase client.

    Returns:
        List[Dict[str, Any]]: List of scan records ordered by created_at descending.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot retrieve recent scans: Supabase is not configured.")
        return []

    try:
        response = (
            active_client.table("scan_history")
            .select("*")
            .order("created_at", desc=True)
            .limit(max(1, limit))
            .execute()
        )
        return response.data or []
    except APIError as api_err:
        logger.error(f"Database error fetching recent scans: {api_err}")
        return []
    except Exception as exc:
        logger.error(f"Unexpected error fetching recent scans: {exc}", exc_info=True)
        return []


def insert_threat_analysis(
    scan_id: str,
    predicted_class: str,
    risk_score: float,
    detected_indicators: Optional[List[str]] = None,
    model_version: str = "v1.0.0",
    client: Optional[Client] = None,
) -> Optional[Dict[str, Any]]:
    """Store machine learning classification results for a specific scan.

    Args:
        scan_id: UUID of the parent scan record from scan_history.
        predicted_class: Classification outcome ('SAFE', 'SUSPICIOUS', 'MALICIOUS').
        risk_score: Numerical risk probability between 0.0 and 1.0 (or 0-100).
        detected_indicators: Optional list of identified malicious features or red flags.
        model_version: Version identifier of the classifier model.
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: The inserted row data, or None on failure.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot insert threat analysis: Supabase is not configured.")
        return None

    # Sanitize and normalize inputs
    norm_class = str(predicted_class).upper().strip()
    if norm_class in {"SAFE", "BENIGN"}:
        normalized_class = "SAFE"
    elif norm_class in {"MALICIOUS"}:
        normalized_class = "MALICIOUS"
    else:
        normalized_class = "SUSPICIOUS"

    clamped_score = round(max(0.0, min(1.0, float(risk_score))), 4)
    indicators = detected_indicators or []

    record = {
        "scan_id": scan_id,
        "predicted_class": normalized_class,
        "risk_score": clamped_score,
        "detected_indicators": indicators,
        "model_version": model_version,
    }

    try:
        response = active_client.table("threat_analysis").insert(record).execute()
        if response.data and len(response.data) > 0:
            inserted = response.data[0]
            logger.info(f"Recorded threat analysis {inserted.get('id')} for scan {scan_id}")
            return inserted
        return None
    except APIError as api_err:
        logger.error(f"Database error inserting threat analysis: {api_err}")
        return None
    except Exception as exc:
        logger.error(f"Unexpected error inserting threat analysis: {exc}", exc_info=True)
        return None


def insert_prevention_decision(
    scan_id: str,
    selected_action: str,
    decision_reason: str,
    policy_version: str = "v1.0.0",
    client: Optional[Client] = None,
) -> Optional[Dict[str, Any]]:
    """Record an adaptive threat prevention decision (ALLOW, WARN_USER, DEEP_INSPECT, BLOCK).

    Args:
        scan_id: UUID of the parent scan record.
        selected_action: Action taken ('ALLOW', 'WARN_USER', 'DEEP_INSPECT', 'BLOCK').
        decision_reason: Human-readable rationale for the action.
        policy_version: Version identifier for the RL policy.
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: The inserted row data, or None on failure.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot insert prevention decision: Supabase is not configured.")
        return None

    normalized_action = selected_action.upper().strip()
    if normalized_action == "WARN":
        normalized_action = "WARN_USER"

    valid_actions = {"ALLOW", "WARN_USER", "DEEP_INSPECT", "BLOCK"}
    if normalized_action not in valid_actions:
        normalized_action = "WARN_USER"

    record = {
        "scan_id": scan_id,
        "selected_action": normalized_action,
        "decision_reason": decision_reason.strip() or "Standard policy evaluation",
        "policy_version": policy_version,
    }

    try:
        response = active_client.table("prevention_decisions").insert(record).execute()
        if response.data and len(response.data) > 0:
            inserted = response.data[0]
            logger.info(f"Recorded prevention decision {inserted.get('id')} ({normalized_action}) for scan {scan_id}")
            return inserted
        return None
    except APIError as api_err:
        logger.error(f"Database error inserting prevention decision: {api_err}")
        return None
    except Exception as exc:
        logger.error(f"Unexpected error inserting prevention decision: {exc}", exc_info=True)
        return None


def insert_rl_feedback(
    scan_id: str,
    state_label: str,
    selected_action: str,
    reward: float,
    feedback_source: str = "USER",
    client: Optional[Client] = None,
) -> Optional[Dict[str, Any]]:
    """Store reinforcement learning feedback for policy optimization.

    Args:
        scan_id: UUID of the parent scan record.
        state_label: Discrete state representation (e.g., 'HIGH_RISK_REDIRECT').
        selected_action: Action taken ('ALLOW', 'WARN_USER', 'DEEP_INSPECT', 'BLOCK').
        reward: Numerical reward value assigned based on outcome.
        feedback_source: Source of feedback ('USER', 'AUTOMATED', 'ADMIN').
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: The inserted row data, or None on failure.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot insert RL feedback: Supabase is not configured.")
        return None

    normalized_action = selected_action.upper().strip()
    if normalized_action == "WARN":
        normalized_action = "WARN_USER"

    valid_actions = {"ALLOW", "WARN_USER", "DEEP_INSPECT", "BLOCK"}
    if normalized_action not in valid_actions:
        normalized_action = "WARN_USER"

    normalized_source = feedback_source.upper().strip()
    valid_sources = {"USER", "AUTOMATED", "ADMIN"}
    if normalized_source not in valid_sources:
        normalized_source = "USER"

    record = {
        "scan_id": scan_id,
        "state_label": state_label.strip(),
        "selected_action": normalized_action,
        "reward": float(reward),
        "feedback_source": normalized_source,
    }

    try:
        response = active_client.table("rl_feedback").insert(record).execute()
        if response.data and len(response.data) > 0:
            inserted = response.data[0]
            logger.info(f"Recorded RL feedback {inserted.get('id')} (Reward: {reward}) for scan {scan_id}")
            return inserted
        return None
    except APIError as api_err:
        logger.error(f"Database error inserting RL feedback: {api_err}")
        return None
    except Exception as exc:
        logger.error(f"Unexpected error inserting RL feedback: {exc}", exc_info=True)
        return None


def save_complete_scan(
    extracted_url: Optional[str],
    qr_payload_type: str,
    predicted_class: str,
    risk_score: float,
    detected_indicators: Optional[List[str]],
    selected_action: str,
    decision_reason: str,
    scan_id: Optional[str] = None,
    model_version: str = "v1.0.0",
    policy_version: str = "v1.0.0",
    client: Optional[Client] = None,
) -> Optional[Dict[str, Any]]:
    """Save complete scan results across scan_history, threat_analysis, and prevention_decisions.

    Enforces idempotency if a scan_id is provided.

    Args:
        extracted_url: Decoded URL or text from QR.
        qr_payload_type: Payload type ('URL', 'TEXT', 'WIFI', 'VCARD', 'OTHER').
        predicted_class: Model predicted class ('SAFE', 'SUSPICIOUS', 'MALICIOUS').
        risk_score: Model-derived risk score.
        detected_indicators: Red flags or feature indicators list.
        selected_action: Final security-enforced action.
        decision_reason: Rationale text.
        scan_id: Optional client UUID for idempotency.
        model_version: Classifier model version.
        policy_version: RL policy version.
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: Consolidated inserted records or None on failure.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot save complete scan: Supabase is not configured.")
        return None

    # 1. Insert scan_history
    scan_record = insert_scan_record(
        extracted_url=extracted_url,
        qr_payload_type=qr_payload_type,
        scan_status="PROCESSED",
        scan_id=scan_id,
        client=active_client,
    )
    if not scan_record:
        logger.error("Failed to insert base scan_history record.")
        return None

    assigned_scan_id = scan_record["id"]

    # 2. Insert threat_analysis (scale percentage 0-100 to 0.0-1.0 if needed)
    score_val = float(risk_score)
    if score_val > 1.0:
        score_val = score_val / 100.0

    threat_record = insert_threat_analysis(
        scan_id=assigned_scan_id,
        predicted_class=predicted_class,
        risk_score=score_val,
        detected_indicators=detected_indicators,
        model_version=model_version,
        client=active_client,
    )

    # 3. Insert prevention_decisions
    decision_record = insert_prevention_decision(
        scan_id=assigned_scan_id,
        selected_action=selected_action,
        decision_reason=decision_reason,
        policy_version=policy_version,
        client=active_client,
    )

    return {
        "scan_id": assigned_scan_id,
        "scan_record": scan_record,
        "threat_record": threat_record,
        "decision_record": decision_record,
    }


def get_detailed_scan_history(
    limit: int = 50,
    client: Optional[Client] = None,
) -> List[Dict[str, Any]]:
    """Retrieve scan history joined with threat analysis and prevention decision details.

    Args:
        limit: Maximum number of scan records to retrieve.
        client: Optional Supabase client.

    Returns:
        List[Dict[str, Any]]: Flattened records suitable for table display and CSV export.
    """
    active_client = client or get_supabase_client()
    if active_client is None:
        logger.warning("Cannot retrieve detailed scans: Supabase is not configured.")
        return []

    try:
        # Perform PostgREST embedded query across relational foreign keys
        response = (
            active_client.table("scan_history")
            .select(
                "id, created_at, extracted_url, qr_payload_type, scan_status, "
                "threat_analysis(predicted_class, risk_score, detected_indicators, model_version), "
                "prevention_decisions(selected_action, decision_reason, policy_version)"
            )
            .order("created_at", desc=True)
            .limit(max(1, limit))
            .execute()
        )
        raw_rows = response.data or []

        flattened = []
        for row in raw_rows:
            threats = row.get("threat_analysis") or []
            threat = threats[0] if isinstance(threats, list) and threats else (threats if isinstance(threats, dict) else {})

            decisions = row.get("prevention_decisions") or []
            decision = decisions[0] if isinstance(decisions, list) and decisions else (decisions if isinstance(decisions, dict) else {})

            action_val = decision.get("selected_action", "N/A")
            if action_val == "WARN_USER":
                action_val = "WARN"

            flattened.append({
                "scan_id": row.get("id"),
                "timestamp": row.get("created_at"),
                "extracted_url": row.get("extracted_url") or "[Empty / None]",
                "payload_type": row.get("qr_payload_type", "OTHER"),
                "status": row.get("scan_status", "PENDING"),
                "predicted_class": threat.get("predicted_class", "N/A"),
                "risk_score": threat.get("risk_score"),
                "model_version": threat.get("model_version", "N/A"),
                "prevention_action": action_val,
                "decision_reason": decision.get("decision_reason", "N/A"),
                "policy_version": decision.get("policy_version", "N/A"),
            })
        return flattened
    except Exception as exc:
        logger.warning(f"Embedded scan history query encountered exception: {exc}. Trying fallback...")
        try:
            simple_scans = get_recent_scans(limit=limit, client=active_client)
            return [
                {
                    "scan_id": s.get("id"),
                    "timestamp": s.get("created_at"),
                    "extracted_url": s.get("extracted_url") or "[Empty / None]",
                    "payload_type": s.get("qr_payload_type", "OTHER"),
                    "status": s.get("scan_status", "PENDING"),
                    "predicted_class": "N/A",
                    "risk_score": None,
                    "model_version": "N/A",
                    "prevention_action": "N/A",
                    "decision_reason": "N/A",
                    "policy_version": "N/A",
                }
                for s in simple_scans
            ]
        except Exception as inner_exc:
            logger.error(f"Fallback scan history fetch failed: {inner_exc}")
            return []


def export_scan_history_to_csv(scans: List[Dict[str, Any]]) -> str:
    """Format scan history records into a clean RFC-4180 CSV string.

    Args:
        scans: List of scan dictionaries.

    Returns:
        str: CSV formatted string ready for file download.
    """
    import csv
    import io

    headers = [
        "scan_id",
        "timestamp",
        "extracted_url",
        "payload_type",
        "status",
        "predicted_class",
        "risk_score",
        "prevention_action",
        "decision_reason",
        "model_version",
        "policy_version",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()

    for row in scans:
        writer.writerow(row)

    return output.getvalue()


class SupabaseManager:
    """Object-oriented wrapper around QRShield Supabase operations."""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Optional[Client] = None

    def connect(self) -> bool:
        """Initialize connection and verify connectivity."""
        self.client = get_supabase_client(self.supabase_url, self.supabase_key)
        if self.client is None:
            return False
        success, _ = test_connection(self.client)
        return success

    def check_connection(self) -> Tuple[bool, str]:
        """Test current connection status."""
        return test_connection(self.client)

    def log_scan(
        self,
        extracted_url: Optional[str],
        qr_payload_type: str = "URL",
        scan_status: str = "PENDING",
        scan_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record a scan event."""
        return insert_scan_record(
            extracted_url,
            qr_payload_type,
            scan_status,
            scan_id=scan_id,
            client=self.client,
        )

    def get_recent_scans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent scans."""
        return get_recent_scans(limit, client=self.client)

    def get_detailed_scan_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve full joined scan history."""
        return get_detailed_scan_history(limit, client=self.client)

    def log_threat_analysis(
        self,
        scan_id: str,
        predicted_class: str,
        risk_score: float,
        detected_indicators: Optional[List[str]] = None,
        model_version: str = "v1.0.0",
    ) -> Optional[Dict[str, Any]]:
        """Record threat analysis."""
        return insert_threat_analysis(
            scan_id,
            predicted_class,
            risk_score,
            detected_indicators,
            model_version,
            client=self.client,
        )

    def log_prevention_decision(
        self,
        scan_id: str,
        selected_action: str,
        decision_reason: str,
        policy_version: str = "v1.0.0",
    ) -> Optional[Dict[str, Any]]:
        """Record prevention decision."""
        return insert_prevention_decision(
            scan_id,
            selected_action,
            decision_reason,
            policy_version,
            client=self.client,
        )

    def log_rl_feedback(
        self,
        scan_id: str,
        state_label: str,
        selected_action: str,
        reward: float,
        feedback_source: str = "USER",
    ) -> Optional[Dict[str, Any]]:
        """Record RL feedback."""
        return insert_rl_feedback(
            scan_id,
            state_label,
            selected_action,
            reward,
            feedback_source,
            client=self.client,
        )

    def save_complete_scan(
        self,
        extracted_url: Optional[str],
        qr_payload_type: str,
        predicted_class: str,
        risk_score: float,
        detected_indicators: Optional[List[str]],
        selected_action: str,
        decision_reason: str,
        scan_id: Optional[str] = None,
        model_version: str = "v1.0.0",
        policy_version: str = "v1.0.0",
    ) -> Optional[Dict[str, Any]]:
        """Record complete scan outcome."""
        return save_complete_scan(
            extracted_url=extracted_url,
            qr_payload_type=qr_payload_type,
            predicted_class=predicted_class,
            risk_score=risk_score,
            detected_indicators=detected_indicators,
            selected_action=selected_action,
            decision_reason=decision_reason,
            scan_id=scan_id,
            model_version=model_version,
            policy_version=policy_version,
            client=self.client,
        )

    def export_csv(self, scans: List[Dict[str, Any]]) -> str:
        """Export scans to CSV format."""
        return export_scan_history_to_csv(scans)
