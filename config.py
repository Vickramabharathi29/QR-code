"""Configuration module for QRShield.

Loads environment variables, manages path references, and defines system-wide
constants for ML classification, Reinforcement Learning, and Supabase integration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODULES_DIR = BASE_DIR / "modules"

# Load local .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

def get_setting(key: str, default: str = "") -> str:
    """Retrieve configuration setting from Streamlit secrets or environment variables.

    Prioritizes:
    1. Streamlit Cloud st.secrets (both flat and nested tables [supabase], [admin], etc.)
    2. os.environ[key] and case variations
    3. Provided default value.
    """
    # 1. Check Streamlit st.secrets safely first if running within Streamlit runtime
    try:
        import streamlit as st
        secrets = getattr(st, "secrets", None)
        if secrets:
            # A. Check direct flat keys
            for k in (key, key.upper(), key.lower()):
                try:
                    if k in secrets and secrets[k] is not None and str(secrets[k]).strip():
                        return str(secrets[k]).strip()
                except Exception:
                    pass

            # B. Check nested sections: [supabase], [SUPABASE], [Supabase]
            for sec_name in ("supabase", "SUPABASE", "Supabase"):
                if sec_name in secrets:
                    sub = secrets[sec_name]
                    if isinstance(sub, dict) or hasattr(sub, "get"):
                        candidates = [
                            key,
                            key.lower(),
                            key.upper(),
                            key.replace("SUPABASE_", "").lower(),
                            key.replace("SUPABASE_", "").upper(),
                        ]
                        if "URL" in key.upper():
                            candidates.extend(["url", "uri", "supabase_url", "project_url", "endpoint"])
                        if "SERVICE_ROLE" in key.upper() or "SERVICE" in key.upper():
                            candidates.extend(["service_role_key", "service_key", "secret_key", "service_role", "service_role_secret"])
                        elif "KEY" in key.upper():
                            candidates.extend(["key", "anon_key", "publishable_key", "api_key", "service_role_key", "service_key", "secret_key"])
                        for c in candidates:
                            try:
                                if c in sub and sub[c] is not None and str(sub[c]).strip():
                                    return str(sub[c]).strip()
                            except Exception:
                                pass

            # C. Check nested sections: [admin], [ADMIN], [Admin], [auth], [AUTH]
            for sec_name in ("admin", "ADMIN", "Admin", "auth", "AUTH"):
                if sec_name in secrets:
                    sub = secrets[sec_name]
                    if isinstance(sub, dict) or hasattr(sub, "get"):
                        candidates = [
                            key,
                            key.lower(),
                            key.upper(),
                            key.replace("ADMIN_", "").lower(),
                            key.replace("ADMIN_", "").upper(),
                        ]
                        if "PASSWORD" in key.upper():
                            candidates.extend(["password", "initial_password", "admin_password", "admin_initial_password"])
                        if "EMAIL" in key.upper():
                            candidates.extend(["email", "admin_email", "user_email"])
                        for c in candidates:
                            try:
                                if c in sub and sub[c] is not None and str(sub[c]).strip():
                                    return str(sub[c]).strip()
                            except Exception:
                                pass
    except Exception:
        pass

    # 2. Check os.environ (exact, uppercase, lowercase)
    for k in (key, key.upper(), key.lower()):
        val = os.getenv(k)
        if val is not None and str(val).strip():
            return str(val).strip()

    return default


def sync_secrets_to_env() -> None:
    """Synchronize Streamlit Cloud st.secrets into os.environ and config module variables.

    Ensures seamless compatibility across all third-party libraries and runtime modules
    on Streamlit Cloud without leaking secrets into git or logs.
    """
    global SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY, ADMIN_EMAIL, ADMIN_INITIAL_PASSWORD

    # Resolve URL
    resolved_url = get_setting("SUPABASE_URL", "https://tjfhuulhzvlkqvjkevki.supabase.co")
    if resolved_url and "your-project-id" not in resolved_url:
        SUPABASE_URL = resolved_url
        os.environ["SUPABASE_URL"] = resolved_url

    # Resolve Service Role Key
    resolved_srv_key = get_setting("SUPABASE_SERVICE_ROLE_KEY", "")
    if not resolved_srv_key:
        resolved_srv_key = get_setting("SUPABASE_SERVICE_KEY", "") or get_setting("SUPABASE_SECRET_KEY", "")
    if resolved_srv_key and "your-supabase" not in resolved_srv_key:
        SUPABASE_SERVICE_ROLE_KEY = resolved_srv_key
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = resolved_srv_key

    # Resolve Supabase Key (anon or service_role)
    resolved_key = get_setting("SUPABASE_KEY", "")
    if not resolved_key:
        resolved_key = resolved_srv_key or get_setting("SUPABASE_ANON_KEY", "")
    if resolved_key and "your-supabase" not in resolved_key:
        SUPABASE_KEY = resolved_key
        os.environ["SUPABASE_KEY"] = resolved_key
        if not SUPABASE_SERVICE_ROLE_KEY and (resolved_key.startswith("sb_secret_") or "service_role" in resolved_key):
            SUPABASE_SERVICE_ROLE_KEY = resolved_key
            os.environ["SUPABASE_SERVICE_ROLE_KEY"] = resolved_key

    # Resolve Admin Email
    resolved_email = get_setting("ADMIN_EMAIL", "Vickramabharathi@gmail.com")
    if resolved_email and "@" in resolved_email:
        ADMIN_EMAIL = resolved_email
        os.environ["ADMIN_EMAIL"] = resolved_email

    # Resolve Admin Password
    resolved_pwd = get_setting("ADMIN_INITIAL_PASSWORD", "Admin#QRShield2026!")
    if not resolved_pwd or resolved_pwd == "Admin#QRShield2026!":
        alt_pwd = get_setting("ADMIN_PASSWORD", "")
        if alt_pwd:
            resolved_pwd = alt_pwd
    if resolved_pwd:
        ADMIN_INITIAL_PASSWORD = resolved_pwd
        os.environ["ADMIN_INITIAL_PASSWORD"] = resolved_pwd


# Initial load from environment or Streamlit secrets
SUPABASE_URL = get_setting("SUPABASE_URL", "https://tjfhuulhzvlkqvjkevki.supabase.co")
SUPABASE_KEY = get_setting("SUPABASE_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = get_setting("SUPABASE_SERVICE_ROLE_KEY", "")

# Administrator Authentication Configuration
ADMIN_EMAIL = get_setting("ADMIN_EMAIL", "Vickramabharathi@gmail.com")
ADMIN_INITIAL_PASSWORD = get_setting("ADMIN_INITIAL_PASSWORD", "Admin#QRShield2026!")

# Execute sync to ensure consistency
sync_secrets_to_env()

# Application Environment
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Image Upload & QR Scanner Settings
MAX_UPLOAD_SIZE_MB = float(os.getenv("MAX_UPLOAD_SIZE_MB", "5.0"))
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# URL Redirection & Static Analysis Settings
URL_TIMEOUT = int(os.getenv("URL_TIMEOUT_SECONDS", "10"))
MAX_REDIRECTS = int(os.getenv("MAX_REDIRECTS", "10"))

# Heuristic Suspicious Indicators
DEFAULT_SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "update", "banking", "secure", "wallet",
    "signin", "confirm", "password", "authenticate", "recovery", "billing",
    "free", "bonus", "reward", "claim", "support", "service", "appleid",
    "paypal", "chase", "wellsfargo", "crypto", "binance", "metamask",
]
DEFAULT_SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "click", "buzz", "fit", "surf"
]
HEURISTIC_RISK_LOW_MAX = 30.0
HEURISTIC_RISK_MED_MAX = 60.0

# Machine Learning Classifier Constants
DEFAULT_RF_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
ML_CLASSES = {
    0: "SAFE",
    1: "MALICIOUS"
}
RISK_THRESHOLD_HIGH = 0.70
RISK_THRESHOLD_MEDIUM = 0.40

# Reinforcement Learning (Q-Learning) Hyperparameters
DEFAULT_Q_TABLE_PATH = MODELS_DIR / "q_table.json"
DEFAULT_RL_METADATA_PATH = MODELS_DIR / "rl_metadata.json"
RL_ACTIONS = ["ALLOW", "WARN", "BLOCK"]
RL_LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "0.1"))
RL_DISCOUNT_FACTOR = float(os.getenv("RL_DISCOUNT_FACTOR", "0.90"))
RL_EPSILON = float(os.getenv("RL_EPSILON", "0.20"))

# Reward matrix for RL state-action outcomes
RL_REWARDS = {
    # (action, is_malicious)
    ("ALLOW", False): +15.0,        # Correctly allowed safe URL
    ("ALLOW", True): -100.0,        # Catastrophic failure: allowed malicious URL
    ("WARN", False): -5.0,          # Minor friction on safe URL
    ("WARN", True): +5.0,           # Warning on suspicious URL
    ("BLOCK", False): -50.0,        # False positive: blocked benign URL
    ("BLOCK", True): +25.0,         # Success: blocked malicious attack
}


def is_supabase_configured() -> bool:
    """Check if Supabase credentials have been supplied via environment variables or secrets.

    Returns:
        bool: True if SUPABASE_URL and either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY are non-empty and non-placeholder.
    """
    url = (SUPABASE_URL or "").strip()
    key = (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY or "").strip()
    if not url or not key:
        return False
    if "your-project-id" in url or "your-supabase" in key:
        return False
    return True


def is_service_role_configured() -> bool:
    """Check if the server-side service_role key is configured for privileged operations.

    Returns:
        bool: True if SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY is provided and not a placeholder.
    """
    key = (SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if not key or "your-supabase" in key:
        return False
    return True


