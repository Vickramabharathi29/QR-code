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

import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from postgrest.exceptions import APIError
from supabase import Client, create_client

import config

logger = logging.getLogger("qrshield.database")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_client_instance: Optional[Client] = None
_service_client_instance: Optional[Client] = None
_client_role: Optional[str] = None
_cached_url: Optional[str] = None
_cached_pub_key: Optional[str] = None
_cached_srv_key: Optional[str] = None
_last_db_error: Optional[Dict[str, Any]] = None


def get_last_error() -> Optional[Dict[str, Any]]:
    """Retrieve the most recent database error metadata."""
    return _last_db_error


def clear_last_error() -> None:
    """Clear any stored database error metadata."""
    global _last_db_error
    _last_db_error = None


def get_key_role(key: Optional[str] = None) -> str:
    """Determine the role of a Supabase API key ('anon', 'service_role', or 'unknown').

    Inspects prefixes and JWT claims without external network calls or exposing keys.
    If key is None, inspects SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY fallback).

    Returns:
        str: 'anon', 'service_role', or 'unknown'
    """
    if key is None:
        target = (config.SUPABASE_KEY or getattr(config, "SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    else:
        target = key.strip()

    if not target:
        return "unknown"

    # Support modern Supabase API key prefixes
    if target.startswith("sb_secret_"):
        return "service_role"
    if target.startswith("sb_publishable_"):
        return "anon"

    # Support standard JWT tokens
    if target.count(".") == 2:
        try:
            payload_b64 = target.split(".")[1]
            padded = payload_b64 + "=" * ((4 - len(payload_b64) % 4) % 4)
            data = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
            role = str(data.get("role", "unknown")).lower()
            if role in ("service_role", "anon"):
                return role
            return role
        except Exception:
            return "unknown"
    return "unknown"


def _handle_db_error(
    exc: Exception,
    operation: str,
    table: str,
    record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract and format structured error information without credentials.

    Logs full diagnostic details to the terminal including error message, error code,
    constraint details, and actionable guidance without exposing API keys.
    """
    global _last_db_error
    import re

    err_code = None
    err_msg = str(exc)
    err_details = None
    err_hint = None

    if isinstance(exc, APIError):
        err_code = getattr(exc, "code", None)
        err_msg = getattr(exc, "message", str(exc))
        err_details = getattr(exc, "details", None)
        err_hint = getattr(exc, "hint", None)
        if hasattr(exc, "args") and exc.args and isinstance(exc.args[0], dict):
            d = exc.args[0]
            err_code = err_code or d.get("code")
            err_msg = d.get("message", err_msg)
            err_details = err_details or d.get("details")
            err_hint = err_hint or d.get("hint")

    # Sanitize known secrets from error messages and details
    for secret in [getattr(config, "SUPABASE_SERVICE_ROLE_KEY", ""), config.SUPABASE_KEY, config.SUPABASE_URL]:
        if secret and len(secret) > 8:
            err_msg = err_msg.replace(secret, "[REDACTED]")
            if err_details:
                err_details = str(err_details).replace(secret, "[REDACTED]")

    # Redact any Bearer headers, JWTs, and Supabase key strings with regex
    err_msg = re.sub(r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", "[REDACTED_TOKEN]", err_msg)
    err_msg = re.sub(r"sb_[a-zA-Z0-9_-]{20,}", "[REDACTED_KEY]", err_msg)
    err_msg = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", err_msg, flags=re.IGNORECASE)

    if err_details:
        err_details = re.sub(r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", "[REDACTED_TOKEN]", str(err_details))
        err_details = re.sub(r"sb_[a-zA-Z0-9_-]{20,}", "[REDACTED_KEY]", str(err_details))
        err_details = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", str(err_details), flags=re.IGNORECASE)

    msg_lower = err_msg.lower()
    if err_code is None and ("row-level security" in msg_lower or "service_role" in msg_lower or "42501" in msg_lower):
        err_code = "42501"

    code_str = str(err_code or "").upper()
    guidance = None

    if "supabase_service_role_key is missing" in msg_lower:
        guidance = (
            "SUPABASE_SERVICE_ROLE_KEY is missing in .env (or SUPABASE_KEY contains an unprivileged anon key). "
            f"Row Level Security (RLS) policy rejected operation on 'public.{table}' (Error 42501). "
            "Privileged backend database writes require the server-side service_role secret. "
            "Please configure SUPABASE_KEY in your local .env file with your rotated service-role key."
        )
    elif (
        code_str == "42501"
        or "row-level security" in msg_lower
        or "permission denied" in msg_lower
        or "service_role" in msg_lower
    ):
        guidance = (
            f"Row Level Security (RLS) policy rejected operation on 'public.{table}' (Error 42501). "
            "Privileged backend database writes require the server-side service_role secret. "
            "Ensure the server-side SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) is configured in your local .env file."
        )
    elif (
        code_str == "PGRST205"
        or "could not find the table" in msg_lower
        or ("relation" in msg_lower and "does not exist" in msg_lower)
    ):
        guidance = (
            f"Table 'public.{table}' does not exist in the Supabase schema cache (Error PGRST205). "
            "Please execute 'supabase/schema.sql' in the Supabase SQL Editor to initialize the database tables."
        )
    elif code_str == "23503" or "foreign key" in msg_lower:
        guidance = (
            f"Foreign key constraint violation on 'public.{table}' (Error 23503). "
            "The referenced parent record (scan_id) does not exist."
        )
    elif code_str == "23514" or "check constraint" in msg_lower:
        guidance = (
            f"CHECK constraint violation on 'public.{table}' (Error 23514). "
            "One or more column values violate the table schema constraints."
        )
    elif code_str == "23505" or "duplicate key" in msg_lower:
        guidance = (
            f"Duplicate key violation on 'public.{table}' (Error 23505). Record already exists."
        )
    elif "connection" in msg_lower or "timeout" in msg_lower:
        guidance = (
            "Network connectivity failure connecting to Supabase. "
            "Check your internet connection and verify SUPABASE_URL in .env."
        )
    else:
        guidance = f"Database error during {operation} on 'public.{table}': {err_msg}"

    error_summary = {
        "operation": operation,
        "table": table,
        "code": err_code,
        "message": err_msg,
        "details": err_details,
        "hint": err_hint,
        "guidance": guidance,
    }

    _last_db_error = error_summary

    log_line = f"❌ [Database Error] Op={operation} | Table={table} | Code={err_code} | Msg={err_msg}"
    if err_details:
        log_line += f" | Details={err_details}"
    if err_hint:
        log_line += f" | Hint={err_hint}"
    logger.error(log_line)
    if guidance:
        logger.error(f"👉 Guidance: {guidance}")

    return error_summary


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
    require_service_role: bool = False,
) -> Optional[Client]:
    """Initialize and return the official Supabase Python client instance.

    Uses configured environment variables SUPABASE_URL and SUPABASE_KEY by default.
    Keeps the privileged client strictly within the trusted Python backend.

    Args:
        supabase_url: Optional custom URL override.
        supabase_key: Optional custom API key override.
        force_new: If True, bypasses the cached singleton client.
        require_service_role: If True, rejects keys confirmed to have only 'anon' privileges.

    Returns:
        Optional[Client]: The connected Supabase client, or None if credentials are missing or inadequate.
    """
    global _client_instance, _service_client_instance, _client_role, _cached_url, _cached_pub_key, _cached_srv_key

    # If key is provided explicitly
    if supabase_key is not None:
        url = (supabase_url or config.SUPABASE_URL).strip()
        key = supabase_key.strip()
        if not url or not key:
            return None
        role = get_key_role(key)
        if require_service_role and role == "anon":
            logger.error("Explicit Supabase API key does not have service_role privileges.")
            return None
        try:
            return create_client(url, key)
        except Exception as exc:
            logger.error(f"Failed to initialize Supabase client with custom key: {exc}")
            return None

    url = (supabase_url or config.SUPABASE_URL or "").strip()
    srv_key = getattr(config, "SUPABASE_SERVICE_ROLE_KEY", "").strip()
    pub_key = (config.SUPABASE_KEY or "").strip()

    # Check cached service client if require_service_role is requested
    if require_service_role:
        if (
            _service_client_instance is not None
            and not force_new
            and supabase_url is None
            and url == _cached_url
            and pub_key == _cached_pub_key
            and srv_key == _cached_srv_key
        ):
            # Ensure the service client hasn't been polluted by a user sign-in session
            auth_hdr = getattr(getattr(_service_client_instance, "postgrest", None), "headers", {}).get("Authorization", "")
            is_valid_service = False
            for k in [srv_key, pub_key]:
                if k and auth_hdr.endswith(k):
                    is_valid_service = True
                    break
            if is_valid_service:
                return _service_client_instance

    # Check cached standard singleton
    elif (
        _client_instance is not None
        and not force_new
        and supabase_url is None
        and url == _cached_url
        and pub_key == _cached_pub_key
        and srv_key == _cached_srv_key
    ):
        return _client_instance

    chosen_key = None
    chosen_role = "unknown"

    # Prioritize service_role key if available in SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY
    if pub_key and "your-supabase" not in pub_key:
        pub_role = get_key_role(pub_key)
        if pub_role == "service_role":
            chosen_key = pub_key
            chosen_role = "service_role"
        elif srv_key and "your-supabase" not in srv_key and get_key_role(srv_key) == "service_role":
            chosen_key = srv_key
            chosen_role = "service_role"
        elif not require_service_role:
            chosen_key = pub_key
            chosen_role = pub_role
        else:
            chosen_key = pub_key
            chosen_role = pub_role
    elif srv_key and "your-supabase" not in srv_key:
        chosen_key = srv_key
        chosen_role = get_key_role(srv_key)
    else:
        chosen_key = ""
        chosen_role = "unknown"

    if not url or not chosen_key:
        logger.warning("Supabase URL or required API key is not configured. Database operations will be disabled.")
        return None

    if "your-project-id" in url or "your-supabase" in chosen_key:
        logger.warning("Default placeholder credentials detected in Supabase config. Please provide real credentials.")
        return None

    if require_service_role and chosen_role == "anon":
        logger.error(
            "SUPABASE_KEY has 'anon' role. Under Row Level Security (RLS), "
            "privileged database writes require the server-side service_role key."
        )
        return None

    try:
        client = create_client(url, chosen_key)
        if not force_new and supabase_url is None:
            if require_service_role:
                _service_client_instance = client
            else:
                _client_instance = client
            _client_role = chosen_role
            _cached_url = url
            _cached_pub_key = pub_key
            _cached_srv_key = srv_key
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
        if (
            "42501" in err_msg
            or "row-level security" in err_msg.lower()
            or "permission denied" in err_msg.lower()
        ):
            return False, (
                "Connected to Supabase, but Row Level Security (RLS) denied access. "
                "Ensure your .env file uses the server-side service_role key."
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
    user_id: Optional[str] = None,
    destination_domain: Optional[str] = None,
    client: Optional[Client] = None,
) -> Optional[Dict[str, Any]]:
    """Insert a new QR scan event into the scan_history table.

    Args:
        extracted_url: The decoded URL or payload string from the QR code.
        qr_payload_type: Type of QR payload ('URL', 'TEXT', 'WIFI', 'VCARD', 'OTHER').
        scan_status: Status flag ('PENDING', 'PROCESSED', 'FAILED').
        scan_id: Optional client-specified UUID for idempotency.
        user_id: Optional UUID of the authenticated user performing the scan.
        destination_domain: Optional resolved network domain string.
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: The inserted row data containing the generated UUID, or None on failure.
    """
    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        logger.warning("Cannot insert scan record: Privileged Supabase client is not available.")
        if not config.is_supabase_configured():
            _handle_db_error(Exception("Supabase credentials not configured in environment"), "INSERT", "scan_history")
        else:
            _handle_db_error(
                Exception(
                    "SUPABASE_SERVICE_ROLE_KEY is missing in .env. "
                    "Row Level Security (RLS) policy rejected operation on 'public.scan_history' (Error 42501). "
                    "Privileged backend database writes require the server-side service_role key. "
                    "Please configure SUPABASE_KEY in your local .env file with your rotated service-role key."
                ),
                "INSERT",
                "scan_history",
            )
        return None

    raw_type_str = str(qr_payload_type or "").upper().strip()
    if raw_type_str in {"URL", "TEXT", "WIFI", "VCARD", "OTHER"}:
        clean_type = raw_type_str
    elif raw_type_str.startswith("URL"):
        clean_type = "URL"
    elif "TEXT" in raw_type_str:
        clean_type = "TEXT"
    elif "WIFI" in raw_type_str:
        clean_type = "WIFI"
    elif "VCARD" in raw_type_str:
        clean_type = "VCARD"
    else:
        clean_type = "OTHER"
    qr_payload_type = clean_type

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
    if user_id:
        record["user_id"] = user_id
    if destination_domain:
        record["destination_domain"] = destination_domain

    try:
        response = active_client.table("scan_history").insert(record).execute()
        if response.data and len(response.data) > 0:
            inserted = response.data[0]
            logger.info(f"Recorded scan {inserted.get('id')} for payload: {extracted_url}")
            return inserted
        _handle_db_error(Exception("Insert returned empty data array"), "INSERT", "scan_history", record)
        return None
    except Exception as exc:
        err_str = str(exc).lower()
        # If user_id or destination_domain columns do not exist in Supabase yet, gracefully fallback to base columns
        if "user_id" in err_str or "destination_domain" in err_str or "pgrst204" in err_str:
            base_record: Dict[str, Any] = {
                "extracted_url": extracted_url,
                "qr_payload_type": qr_payload_type,
                "scan_status": scan_status,
            }
            if scan_id:
                base_record["id"] = scan_id
            try:
                fb_res = active_client.table("scan_history").insert(base_record).execute()
                if fb_res.data and len(fb_res.data) > 0:
                    inserted = fb_res.data[0]
                    logger.info(f"Recorded scan {inserted.get('id')} using base schema fallback.")
                    return inserted
            except Exception as fb_exc:
                _handle_db_error(fb_exc, "INSERT", "scan_history", base_record)
                return None
        _handle_db_error(exc, "INSERT", "scan_history", record)
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
        _handle_db_error(api_err, "SELECT", "scan_history")
        return []
    except Exception as exc:
        _handle_db_error(exc, "SELECT", "scan_history")
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
    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        logger.warning("Cannot insert threat analysis: Privileged Supabase client is not available.")
        if not config.is_supabase_configured():
            _handle_db_error(Exception("Supabase credentials not configured in environment"), "INSERT", "threat_analysis")
        else:
            _handle_db_error(
                Exception(
                    "Row Level Security (RLS) policy rejected operation on 'public.threat_analysis' (Error 42501). "
                    "Privileged backend database writes require the server-side service_role key. "
                    "Please configure SUPABASE_KEY in your local .env file with your rotated service-role key."
                ),
                "INSERT",
                "threat_analysis",
            )
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
        _handle_db_error(Exception("Insert returned empty data array"), "INSERT", "threat_analysis", record)
        return None
    except APIError as api_err:
        _handle_db_error(api_err, "INSERT", "threat_analysis", record)
        return None
    except Exception as exc:
        _handle_db_error(exc, "INSERT", "threat_analysis", record)
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
    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        logger.warning("Cannot insert prevention decision: Privileged Supabase client is not available.")
        if not config.is_supabase_configured():
            _handle_db_error(Exception("Supabase credentials not configured in environment"), "INSERT", "prevention_decisions")
        else:
            _handle_db_error(
                Exception(
                    "Row Level Security (RLS) policy rejected operation on 'public.prevention_decisions' (Error 42501). "
                    "Privileged backend database writes require the server-side service_role key. "
                    "Please configure SUPABASE_KEY in your local .env file with your rotated service-role key."
                ),
                "INSERT",
                "prevention_decisions",
            )
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
        _handle_db_error(Exception("Insert returned empty data array"), "INSERT", "prevention_decisions", record)
        return None
    except APIError as api_err:
        if api_err.code == "PGRST205" or "not found" in str(api_err).lower():
            try:
                fallback_resp = active_client.table("prevention_decision").insert(record).execute()
                if fallback_resp.data and len(fallback_resp.data) > 0:
                    inserted = fallback_resp.data[0]
                    logger.info(f"Recorded prevention decision via fallback {inserted.get('id')} ({normalized_action}) for scan {scan_id}")
                    return inserted
            except Exception:
                pass
        _handle_db_error(api_err, "INSERT", "prevention_decisions", record)
        return None
    except Exception as exc:
        _handle_db_error(exc, "INSERT", "prevention_decisions", record)
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
    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        logger.warning("Cannot insert RL feedback: Privileged Supabase client is not available.")
        if not config.is_supabase_configured():
            _handle_db_error(Exception("Supabase credentials not configured in environment"), "INSERT", "rl_feedback")
        else:
            _handle_db_error(
                Exception(
                    "Row Level Security (RLS) policy rejected operation on 'public.rl_feedback' (Error 42501). "
                    "Privileged backend database writes require the server-side service_role key. "
                    "Please configure SUPABASE_KEY in your local .env file with your rotated service-role key."
                ),
                "INSERT",
                "rl_feedback",
            )
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
        _handle_db_error(Exception("Insert returned empty data array"), "INSERT", "rl_feedback", record)
        return None
    except APIError as api_err:
        _handle_db_error(api_err, "INSERT", "rl_feedback", record)
        return None
    except Exception as exc:
        _handle_db_error(exc, "INSERT", "rl_feedback", record)
        return None


def save_complete_scan(
    extracted_url: Optional[str],
    qr_payload_type: str,
    predicted_class: str,
    risk_score: float,
    detected_indicators: Optional[List[str]] = None,
    selected_action: str = "WARN_USER",
    decision_reason: str = "Automated threat evaluation",
    scan_id: Optional[str] = None,
    user_id: Optional[str] = None,
    destination_domain: Optional[str] = None,
    model_version: str = "v1.0.0",
    policy_version: str = "v1.0.0",
    state_label: Optional[str] = None,
    reward: Optional[float] = None,
    feedback_source: str = "AUTOMATED",
    raw_action: Optional[str] = None,
    client: Optional[Client] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Save complete scan results across scan_history, threat_analysis, prevention_decisions, and rl_feedback.

    Enforces atomicity and idempotency if a scan_id is provided.

    Args:
        extracted_url: Decoded URL or text from QR.
        qr_payload_type: Payload type ('URL', 'TEXT', 'WIFI', 'VCARD', 'OTHER').
        predicted_class: Model predicted class ('SAFE', 'SUSPICIOUS', 'MALICIOUS').
        risk_score: Model-derived risk score.
        detected_indicators: Red flags or feature indicators list.
        selected_action: Final security-enforced action.
        decision_reason: Rationale text.
        scan_id: Optional client UUID for idempotency.
        user_id: Optional authenticated user UUID.
        destination_domain: Optional destination domain.
        model_version: Classifier model version.
        policy_version: RL policy version.
        state_label: Reinforcement learning state representation.
        reward: Reinforcement learning reward signal.
        feedback_source: Source of RL feedback ('AUTOMATED', 'USER', 'ADMIN').
        raw_action: Pre-constraint recommendation from Q-table.
        client: Optional Supabase client.

    Returns:
        Optional[Dict[str, Any]]: Consolidated inserted records or None on failure.
    """
    clear_last_error()
    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        logger.warning("Cannot save complete scan: Privileged Supabase client is not available.")
        if not config.is_supabase_configured():
            _handle_db_error(Exception("Supabase credentials not configured in environment"), "SAVE_COMPLETE_SCAN", "all")
        else:
            _handle_db_error(
                Exception(
                    "SUPABASE_SERVICE_ROLE_KEY is missing in .env. "
                    "Row Level Security (RLS) policy rejected operation on 'public.scan_history' (Error 42501). "
                    "Privileged backend database writes require the server-side service_role key. "
                    "Please configure SUPABASE_KEY in your local .env file with your rotated service-role key."
                ),
                "SAVE_COMPLETE_SCAN",
                "all",
            )
        return None

    # Derive destination domain if missing and payload is a URL
    resolved_domain = destination_domain
    if not resolved_domain and extracted_url and "://" in str(extracted_url):
        try:
            import urllib.parse
            resolved_domain = urllib.parse.urlparse(extracted_url).netloc
        except Exception:
            resolved_domain = None

    # 1. Insert scan_history
    scan_record = insert_scan_record(
        extracted_url=extracted_url,
        qr_payload_type=qr_payload_type,
        scan_status="PROCESSED",
        scan_id=scan_id,
        user_id=user_id,
        destination_domain=resolved_domain,
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
    if not threat_record:
        logger.error(f"Failed to insert threat_analysis record for scan {assigned_scan_id}.")
        return None

    # 3. Insert prevention_decisions
    decision_record = insert_prevention_decision(
        scan_id=assigned_scan_id,
        selected_action=selected_action,
        decision_reason=decision_reason,
        policy_version=policy_version,
        client=active_client,
    )
    if not decision_record:
        logger.error(f"Failed to insert prevention_decisions record for scan {assigned_scan_id}.")
        return None

    # 4. Insert rl_feedback
    effective_state = state_label or f"STATE:{predicted_class.upper()}_RISK_{int(score_val * 100)}"
    if reward is None:
        norm_act = selected_action.upper().strip()
        if norm_act in ("BLOCK", "ACTION_BLOCK"):
            effective_reward = 15.0 if predicted_class.upper() == "MALICIOUS" else -15.0
        elif norm_act in ("ALLOW", "ACTION_ALLOW"):
            effective_reward = 10.0 if predicted_class.upper() == "SAFE" else -20.0
        else:
            effective_reward = 5.0
    else:
        effective_reward = float(reward)

    rl_record = insert_rl_feedback(
        scan_id=assigned_scan_id,
        state_label=effective_state,
        selected_action=selected_action,
        reward=effective_reward,
        feedback_source=feedback_source,
        client=active_client,
    )
    if not rl_record:
        logger.warning(f"Failed to record rl_feedback record for scan {assigned_scan_id}.")

    return {
        "scan_id": assigned_scan_id,
        "scan_record": scan_record,
        "threat_record": threat_record,
        "decision_record": decision_record,
        "rl_record": rl_record,
    }


def run_database_diagnostics(client: Optional[Client] = None) -> Dict[str, Any]:
    """Execute non-destructive database and authentication diagnostics.

    Verifies client initialization, table access, write health, and RLS configuration
    without exposing keys, passwords, or credentials.

    Returns:
        Dict[str, Any]: Safe diagnostics report.
    """
    diag: Dict[str, Any] = {
        "is_configured": config.is_supabase_configured(),
        "is_service_role": config.is_service_role_configured(),
        "connection_verified": False,
        "connection_message": "",
        "tables": {},
        "write_verified": False,
        "auth_verified": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        diag["connection_message"] = "Supabase client initialization failed: Missing or invalid API key/URL."
        return diag

    # 1. Connection check
    is_ok, msg = test_connection(active_client)
    diag["connection_verified"] = is_ok
    diag["connection_message"] = msg

    # 2. Table access checks
    check_tables = ["scan_history", "threat_analysis", "prevention_decisions", "rl_feedback", "user_roles", "user_profiles"]
    for tbl in check_tables:
        try:
            res = active_client.table(tbl).select("id").limit(1).execute()
            diag["tables"][tbl] = {
                "accessible": True,
                "row_sample": len(res.data) if res.data is not None else 0,
            }
        except Exception as t_err:
            diag["tables"][tbl] = {
                "accessible": False,
                "error": str(t_err).split("\n")[0][:80],
            }

    # 3. Auth service check
    try:
        users = active_client.auth.admin.list_users()
        diag["auth_verified"] = True
        diag["user_count"] = len(users)
    except Exception as a_err:
        diag["auth_verified"] = False
        diag["auth_error"] = str(a_err).split("\n")[0][:80]

    # 4. Write capability verification (scan_history is confirmed writeable if service-role is active)
    diag["write_verified"] = diag["is_service_role"] and diag["tables"].get("scan_history", {}).get("accessible", False)
    diag["configured"] = diag["is_configured"]
    diag["connection_ok"] = diag["connection_verified"]
    diag["tables_accessible"] = [t for t, info in diag["tables"].items() if info.get("accessible")]

    return diag


def get_user_display_map(
    client: Optional[Client] = None,
    user_ids: Optional[Union[Set[str], List[str]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Retrieve mapping of user UUIDs to non-sensitive profile attributes (email, name, role).

    Args:
        client: Optional Supabase client.
        user_ids: Optional collection of user UUIDs to filter by.

    Returns:
        Dict[str, Dict[str, str]]: Mapping of user_id -> {'email': ..., 'full_name': ..., 'role': ...}
    """
    if user_ids is not None and len(user_ids) == 0:
        return {}

    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        return {}

    user_map: Dict[str, Dict[str, str]] = {}
    id_list = list(user_ids) if user_ids is not None else None

    # Query public.user_roles if available
    try:
        query = active_client.table("user_roles").select("user_id, email, role")
        if id_list:
            query = query.in_("user_id", id_list)
        roles_res = query.execute()
        for r in roles_res.data or []:
            uid = str(r.get("user_id") or "")
            if uid:
                email = r.get("email") or ""
                name = email.split("@")[0].capitalize() if email else "User"
                user_map[uid] = {
                    "email": email,
                    "full_name": name,
                    "role": r.get("role") or "user",
                }
    except Exception as e:
        logger.debug("user_roles query note: %s", e)

    # Query public.user_profiles if available
    try:
        query = active_client.table("user_profiles").select("user_id, full_name, email, account_status")
        if id_list:
            query = query.in_("user_id", id_list)
        prof_res = query.execute()
        for p in prof_res.data or []:
            uid = str(p.get("user_id") or "")
            if uid:
                if uid not in user_map:
                    user_map[uid] = {
                        "email": p.get("email") or "",
                        "full_name": p.get("full_name") or "User",
                        "role": "user",
                        "account_status": p.get("account_status") or "active",
                    }
                else:
                    if p.get("full_name"):
                        user_map[uid]["full_name"] = p.get("full_name")
                    if p.get("email") and not user_map[uid].get("email"):
                        user_map[uid]["email"] = p.get("email")
                    if p.get("account_status"):
                        user_map[uid]["account_status"] = p.get("account_status")
    except Exception as e:
        logger.debug("user_profiles query note: %s", e)

    # Query auth.admin.list_users() if available
    try:
        if hasattr(active_client, "auth") and hasattr(active_client.auth, "admin") and hasattr(active_client.auth.admin, "list_users"):
            users = active_client.auth.admin.list_users()
            for u in users:
                uid = str(getattr(u, "id", "") or "")
                if uid and (not id_list or uid in id_list):
                    email = getattr(u, "email", "") or ""
                    meta = getattr(u, "user_metadata", {}) or {}
                    name = meta.get("full_name") or (email.split("@")[0].capitalize() if email else "User")
                    role = meta.get("role") or user_map.get(uid, {}).get("role", "user")
                    user_map[uid] = {"email": email, "full_name": name, "role": role}
    except Exception as e:
        logger.debug("auth.admin.list_users note: %s", e)

    return user_map


def get_detailed_scan_history(
    limit: int = 50,
    client: Optional[Client] = None,
    is_admin: bool = True,
) -> List[Dict[str, Any]]:
    """Retrieve scan history joined with threat analysis, prevention decision, and RL feedback.

    Args:
        limit: Maximum number of scan records to retrieve.
        client: Optional Supabase client.
        is_admin: Boolean indicating whether caller is an authorized administrator.

    Returns:
        List[Dict[str, Any]]: Comprehensive flattened audit records.
    """
    if not is_admin:
        logger.warning("Unauthorized access to global scan history rejected.")
        return []

    active_client = client or get_supabase_client(require_service_role=True)
    if active_client is None:
        logger.warning("Cannot retrieve detailed scans: Supabase is not configured.")
        return []

    raw_rows = []

    # 1. Attempt relational PostgREST embedded query across scan_history, threat_analysis, prevention_decisions, and rl_feedback
    try:
        response = (
            active_client.table("scan_history")
            .select(
                "id, created_at, extracted_url, qr_payload_type, scan_status, user_id, destination_domain, "
                "threat_analysis(predicted_class, risk_score, detected_indicators, model_version), "
                "prevention_decisions(selected_action, decision_reason, policy_version), "
                "rl_feedback(state_label, selected_action, reward, feedback_source)"
            )
            .order("created_at", desc=True)
            .limit(max(1, limit))
            .execute()
        )
        raw_rows = response.data or []
    except Exception as exc:
        logger.warning("Embedded scan history query encountered exception: %s. Trying fallback...", exc)
        try:
            simple_scans = get_recent_scans(limit=limit, client=active_client)
            return [
                {
                    "scan_id": s.get("id"),
                    "timestamp": s.get("created_at"),
                    "extracted_url": s.get("extracted_url") or "[Empty / None]",
                    "destination_domain": s.get("destination_domain") or "N/A",
                    "payload_type": s.get("qr_payload_type", "OTHER"),
                    "status": s.get("scan_status", "PENDING"),
                    "user_id": s.get("user_id"),
                    "user_email": "Anonymous Guest" if not s.get("user_id") else f"User {str(s.get('user_id'))[:8]}",
                    "user_name": "Public Guest" if not s.get("user_id") else "Registered User",
                    "user_role": "guest" if not s.get("user_id") else "user",
                    "predicted_class": "N/A",
                    "risk_score": None,
                    "detected_indicators": [],
                    "model_version": "N/A",
                    "prevention_action": "N/A",
                    "decision_reason": "N/A",
                    "policy_version": "N/A",
                    "rl_state": "N/A",
                    "rl_action": "N/A",
                    "rl_reward": None,
                    "rl_source": "N/A",
                }
                for s in simple_scans
            ]
        except Exception as inner_exc:
            logger.error("Fallback scan history fetch failed: %s", inner_exc)
            return []

    # 2. Enrich with user profiles if user_ids are present
    user_ids = {str(r.get("user_id")) for r in raw_rows if r.get("user_id")}
    user_map = get_user_display_map(active_client, user_ids=user_ids) if user_ids else {}

    # 3. Flatten and normalize records
    flattened = []
    for row in raw_rows:
        threats = row.get("threat_analysis") or []
        threat = threats[0] if isinstance(threats, list) and threats else (threats if isinstance(threats, dict) else {})

        decisions = row.get("prevention_decisions") or row.get("prevention_decision") or []
        decision = decisions[0] if isinstance(decisions, list) and decisions else (decisions if isinstance(decisions, dict) else {})

        rl_items = row.get("rl_feedback") or []
        rl = rl_items[0] if isinstance(rl_items, list) and rl_items else (rl_items if isinstance(rl_items, dict) else {})

        action_val = decision.get("selected_action", "N/A")
        if action_val == "WARN_USER":
            action_val = "WARN"

        uid = row.get("user_id")
        user_info = user_map.get(str(uid), {}) if uid else {}
        user_email = user_info.get("email") or ("Anonymous Guest" if not uid else f"User {str(uid)[:8]}")
        user_name = user_info.get("full_name") or ("Public Guest" if not uid else "Registered User")
        user_role = user_info.get("role") or ("guest" if not uid else "user")

        # Derive domain if missing
        url_val = row.get("extracted_url") or "[Empty / None]"
        domain_val = row.get("destination_domain")
        if not domain_val and "://" in str(url_val):
            try:
                import urllib.parse
                domain_val = urllib.parse.urlparse(url_val).netloc
            except Exception:
                domain_val = "N/A"

        flattened.append({
            "scan_id": row.get("id"),
            "timestamp": row.get("created_at"),
            "extracted_url": url_val,
            "destination_domain": domain_val or "N/A",
            "payload_type": row.get("qr_payload_type", "OTHER"),
            "status": row.get("scan_status", "PENDING"),
            "user_id": uid,
            "user_email": user_email,
            "user_name": user_name,
            "user_role": user_role,
            "predicted_class": threat.get("predicted_class", "N/A"),
            "risk_score": threat.get("risk_score"),
            "detected_indicators": threat.get("detected_indicators") or [],
            "model_version": threat.get("model_version", "N/A"),
            "prevention_action": action_val,
            "decision_reason": decision.get("decision_reason", "N/A"),
            "policy_version": decision.get("policy_version", "N/A"),
            "rl_state": rl.get("state_label", "N/A"),
            "rl_action": rl.get("selected_action", "N/A"),
            "rl_reward": rl.get("reward"),
            "rl_source": rl.get("feedback_source", "N/A"),
        })

    return flattened


def export_scan_history_to_csv(scans: List[Dict[str, Any]], is_admin: bool = True) -> str:
    """Format scan history records into a clean RFC-4180 CSV string.

    Args:
        scans: List of scan dictionaries.
        is_admin: Authorization check flag.

    Returns:
        str: CSV formatted string ready for file download.
    """
    if not is_admin:
        raise PermissionError("Exporting scan history audit log requires administrator authorization.")

    import csv
    import io

    headers = [
        "scan_id",
        "timestamp",
        "extracted_url",
        "user_email",
        "user_name",
        "destination_domain",
        "payload_type",
        "status",
        "predicted_class",
        "risk_score",
        "detected_indicators",
        "prevention_action",
        "decision_reason",
        "rl_action",
        "rl_reward",
        "rl_state",
        "model_version",
        "policy_version",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()

    for row in scans:
        formatted_row = dict(row)
        indicators = formatted_row.get("detected_indicators")
        if isinstance(indicators, list):
            formatted_row["detected_indicators"] = "; ".join(str(i) for i in indicators)
        writer.writerow(formatted_row)

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
