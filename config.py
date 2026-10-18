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

# Supabase PostgreSQL Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://tjfhuulhzvlkqvjkevki.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Administrator Authentication Configuration
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "Vickramabharathi@gmail.com")
ADMIN_INITIAL_PASSWORD = os.getenv("ADMIN_INITIAL_PASSWORD", "Admin#QRShield2026!")


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
    """Check if Supabase credentials have been supplied via environment variables.

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

