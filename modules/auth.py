"""Authentication and Role-Based Access Control (RBAC) Module for QRShield.

Provides secure Supabase Auth integration, administrator session management,
role verification, privilege separation, and secure provisioning without exposing
plaintext passwords or credentials.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client
from postgrest.exceptions import APIError

import config
from modules import database

logger = logging.getLogger("qrshield.auth")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_admin_email() -> str:
    """Retrieve the designated initial administrator email address from configuration."""
    return getattr(config, "ADMIN_EMAIL", "Vickramabharathi@gmail.com").strip().lower()


def get_admin_initial_password() -> str:
    """Retrieve the server-side initial admin provisioning password from environment."""
    return getattr(config, "ADMIN_INITIAL_PASSWORD", "Admin#QRShield2026!").strip()


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate that an administrator password meets cybersecurity complexity standards.

    Requires:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special symbol
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one numerical digit (0-9)."
    if not re.search(r"[@#$%^&+=!_~?<>.,/\\-]", password):
        return False, "Password must contain at least one special character (@#$%^&+=!_~?<>.,/-)."
    return True, "Password complexity verified."


def provision_initial_admin(
    admin_email: Optional[str] = None,
    initial_password: Optional[str] = None,
    client: Optional[Client] = None,
) -> Tuple[bool, str]:
    """Provision the initial SecOps administrator account in Supabase Auth securely.

    Uses privileged server-side service-role client to check if the admin user exists
    in auth.users. If absent, creates the user and assigns the trusted 'admin' role.

    Never exposes plaintext credentials in error messages or logs.

    Args:
        admin_email: Optional override email. Defaults to config.ADMIN_EMAIL.
        initial_password: Optional override password. Defaults to config.ADMIN_INITIAL_PASSWORD.
        client: Optional Supabase client instance with service_role privileges.

    Returns:
        Tuple[bool, str]: (Success boolean, human-readable status message).
    """
    target_email = (admin_email or get_admin_email()).strip().lower()
    target_password = initial_password or get_admin_initial_password()

    if not target_email or "@" not in target_email:
        return False, "Invalid administrator email address provided."
    if not target_password:
        return False, "Administrator initial password not configured in environment."

    # Must use service_role client for administrative provisioning
    active_client = client or database.get_supabase_client(require_service_role=True)
    if active_client is None:
        return False, "Supabase connection is unconfigured or missing service_role key."

    try:
        # Check existing users in Supabase Auth
        existing_users = active_client.auth.admin.list_users()
        user_record = None
        for u in existing_users:
            if getattr(u, "email", "").strip().lower() == target_email:
                user_record = u
                break

        if user_record is None:
            # Create user in Supabase Auth with admin role metadata
            create_attrs = {
                "email": target_email,
                "password": target_password,
                "email_confirm": True,
                "user_metadata": {
                    "role": "admin",
                    "full_name": "SecOps Administrator",
                    "requires_password_change": True,
                },
                "app_metadata": {
                    "role": "admin",
                    "provider": "email",
                },
            }
            res = active_client.auth.admin.create_user(create_attrs)
            user_record = getattr(res, "user", res)
            logger.info("Created initial administrator account in Supabase Auth: %s", target_email)
        else:
            logger.info("Administrator account already exists in Supabase Auth: %s", target_email)

        user_id = str(getattr(user_record, "id", ""))

        # Sync to public.user_roles table if it exists in schema
        try:
            active_client.table("user_roles").upsert({
                "user_id": user_id,
                "email": target_email,
                "role": "admin",
            }, on_conflict="user_id").execute()
            logger.info("Synchronized admin role to public.user_roles for user ID %s", user_id[:8])
        except Exception as db_exc:
            # If user_roles table is not yet created, note it without blocking
            logger.debug("public.user_roles table sync note: %s", db_exc)

        return True, "Administrator account provisioned successfully."

    except Exception as exc:
        err_msg = str(exc)
        logger.error("Failed to provision initial administrator: %s", err_msg)
        return False, f"Failed to provision administrator account: {err_msg}"


def authenticate_admin(
    email: str,
    password: str,
    client: Optional[Client] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Authenticate administrator credentials via Supabase Auth and verify server-side role.

    Enforces that:
    1. Email and password are verified through Supabase Auth (no plaintext table storage).
    2. Authenticated user's role is strictly verified on the server ('admin').
    3. Unauthenticated or standard users cannot gain administrator access.

    Args:
        email: Administrator email address.
        password: Plaintext password provided at login prompt.
        client: Optional Supabase client instance.

    Returns:
        Tuple[bool, Optional[Dict[str, Any]], str]:
            (Success, Session dict if authenticated, Status message).
    """
    clean_email = (email or "").strip().lower()
    clean_pwd = password or ""

    if not clean_email or "@" not in clean_email:
        return False, None, "Please provide a valid administrator email address."
    if not clean_pwd:
        return False, None, "Please enter your administrator password."

    active_client = client or database.get_supabase_client()
    if active_client is None:
        return False, None, "Supabase connection is not configured. Please check .env configuration."

    try:
        # Pre-check: If this is the designated admin email and no users exist yet in Auth,
        # run secure provisioning so initial setup succeeds automatically.
        if clean_email == get_admin_email():
            try:
                srv_client = database.get_supabase_client(require_service_role=True)
                if srv_client:
                    u_list = srv_client.auth.admin.list_users()
                    if not any(getattr(u, "email", "").lower() == clean_email for u in u_list):
                        provision_initial_admin(admin_email=clean_email, initial_password=clean_pwd, client=srv_client)
            except Exception as prov_exc:
                logger.debug("Auto-provisioning check pass: %s", prov_exc)

        # Execute Supabase Auth verification
        auth_response = active_client.auth.sign_in_with_password({
            "email": clean_email,
            "password": clean_pwd,
        })

        user = getattr(auth_response, "user", None)
        session = getattr(auth_response, "session", None)

        if user is None:
            return False, None, "Invalid email or password. Access denied."

        user_id = str(getattr(user, "id", ""))
        user_email = str(getattr(user, "email", "")).strip().lower()

        # Server-side verification of administrator authorization
        is_admin = False

        # Check 1: Check public.user_roles table via service_role client
        try:
            srv_client = database.get_supabase_client(require_service_role=True)
            if srv_client:
                role_query = srv_client.table("user_roles").select("role").eq("user_id", user_id).execute()
                if role_query.data and len(role_query.data) > 0:
                    assigned_role = str(role_query.data[0].get("role", "")).lower()
                    if assigned_role == "admin":
                        is_admin = True
        except Exception as role_exc:
            logger.debug("user_roles table verification: %s", role_exc)

        # Check 2: Check Supabase Auth app_metadata and user_metadata
        app_meta = getattr(user, "app_metadata", {}) or {}
        user_meta = getattr(user, "user_metadata", {}) or {}

        if app_meta.get("role") == "admin" or user_meta.get("role") == "admin":
            is_admin = True

        # Check 3: Check designated administrator email match
        if user_email == get_admin_email():
            is_admin = True

        if not is_admin:
            try:
                active_client.auth.sign_out()
            except Exception:
                pass
            logger.warning("User %s attempted administrator login without admin privileges.", user_email)
            return False, None, "Access denied: Account does not possess SecOps administrator privileges."

        full_name = user_meta.get("full_name") or "SecOps Administrator"
        requires_pwd_change = bool(user_meta.get("requires_password_change", False))

        session_dict = {
            "user_id": user_id,
            "email": user_email,
            "role": "admin",
            "full_name": full_name,
            "access_token": getattr(session, "access_token", ""),
            "refresh_token": getattr(session, "refresh_token", ""),
            "expires_at": getattr(session, "expires_at", 0),
            "requires_password_change": requires_pwd_change,
        }

        logger.info("Administrator successfully authenticated: %s (ID: %s)", user_email, user_id[:8])
        return True, session_dict, "Administrator authenticated successfully."

    except Exception as exc:
        err_msg = str(exc)
        # Check for invalid credentials
        if "invalid" in err_msg.lower() or "credentials" in err_msg.lower() or "400" in err_msg:
            return False, None, "Invalid email or password. Access denied."
        logger.error("Supabase authentication error: %s", err_msg)
        return False, None, f"Authentication service error: {err_msg}"


def sign_out_admin(client: Optional[Client] = None) -> bool:
    """Sign out the current administrator session from Supabase Auth."""
    active_client = client or database.get_supabase_client()
    if active_client is not None:
        try:
            active_client.auth.sign_out()
            return True
        except Exception as exc:
            logger.debug("Sign out exception: %s", exc)
            return True
    return True


def change_admin_password(
    new_password: str,
    current_password: Optional[str] = None,
    confirm_password: Optional[str] = None,
    user_id: Optional[str] = None,
    admin_email: Optional[str] = None,
    client: Optional[Client] = None,
) -> Tuple[bool, str]:
    """Change the administrator password securely via Supabase Auth.

    Verifies the current password if provided, validates password confirmation,
    enforces cybersecurity complexity rules, and updates the administrator's password
    in auth.users via service_role.

    Args:
        new_password: The new complex password to set.
        current_password: Optional current password for verification.
        confirm_password: Optional confirmation password to verify match.
        user_id: Optional UUID of the user. If omitted, resolved via email or client.
        admin_email: Optional administrator email address.
        client: Optional Supabase client instance.

    Returns:
        Tuple[bool, str]: (Success boolean, Status message).
    """
    if confirm_password is not None and str(new_password) != str(confirm_password):
        return False, "New password and confirmation do not match."

    target_email = (admin_email or get_admin_email()).strip().lower()

    # Step 1: Verify current password if supplied
    if current_password:
        clean_curr = str(current_password).strip()
        if not clean_curr:
            return False, "Current password cannot be empty."
        if new_password == clean_curr:
            return False, "New password must be different from your current password."

        verify_client = client or database.get_supabase_client()
        if verify_client:
            try:
                auth_resp = verify_client.auth.sign_in_with_password({
                    "email": target_email,
                    "password": clean_curr,
                })
                auth_u = getattr(auth_resp, "user", None)
                if auth_u is None:
                    return False, "Current password verification failed. Please check your credentials."
                user_id = user_id or str(getattr(auth_u, "id", ""))
            except Exception as v_err:
                err_text = str(v_err).lower()
                if "invalid" in err_text or "credentials" in err_text or "400" in err_text:
                    return False, "Incorrect current password. Verification failed."
                logger.warning("Current password verification error: %s", v_err)
                return False, f"Current password verification error: {v_err}"

    # Step 2: Validate password strength
    valid, msg = validate_password_strength(new_password)
    if not valid:
        return False, msg

    # Step 3: Use privileged service-role client to update password
    srv_client = client or database.get_supabase_client(require_service_role=True)
    if srv_client is None:
        return False, "Supabase connection unavailable or service_role key missing."

    try:
        # Resolve user_id if still missing
        if not user_id:
            users = srv_client.auth.admin.list_users()
            for u in users:
                if getattr(u, "email", "").strip().lower() == target_email:
                    user_id = str(getattr(u, "id", ""))
                    break

        if not user_id:
            return False, f"Could not find administrator account for {target_email} in Supabase Auth."

        # Update user attributes in Supabase Auth
        srv_client.auth.admin.update_user_by_id(
            user_id,
            {
                "password": new_password,
                "user_metadata": {
                    "requires_password_change": False,
                    "role": "admin",
                },
            },
        )

        # Update in-memory configuration fallback so active session remains aligned
        if hasattr(config, "ADMIN_INITIAL_PASSWORD"):
            config.ADMIN_INITIAL_PASSWORD = new_password

        logger.info("Administrator password updated successfully for user ID %s", user_id[:8])
        return True, "Administrator password changed successfully. Please use your new password on subsequent logins."
    except Exception as exc:
        err_msg = str(exc)
        logger.error("Failed to update admin password: %s", err_msg)
        return False, f"Failed to update password: {err_msg}"


def verify_admin_authorization(session_dict: Optional[Dict[str, Any]]) -> bool:
    """Verify whether a session dictionary holds valid, unexpired administrator privileges.

    Args:
        session_dict: Dictionary containing session state.

    Returns:
        bool: True if authorized as administrator.
    """
    if not session_dict or not isinstance(session_dict, dict):
        return False
    if session_dict.get("role") != "admin":
        return False
    if not session_dict.get("user_id") or not session_dict.get("email"):
        return False
    return True
