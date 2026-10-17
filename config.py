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
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Application Environment
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# URL Redirection Analysis Settings
URL_TIMEOUT = int(os.getenv("URL_TIMEOUT_SECONDS", "10"))
MAX_REDIRECTS = int(os.getenv("MAX_REDIRECTS", "10"))

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
RL_ACTIONS = ["ALLOW", "WARN_USER", "DEEP_INSPECT", "BLOCK"]
RL_LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "0.1"))
RL_DISCOUNT_FACTOR = float(os.getenv("RL_DISCOUNT_FACTOR", "0.95"))
RL_EPSILON = float(os.getenv("RL_EPSILON", "0.10"))

# Reward matrix for RL state-action outcomes
RL_REWARDS = {
    # (action, true_threat)
    ("ALLOW", False): +1.0,         # Correctly allowed safe QR
    ("ALLOW", True): -10.0,         # Catastrophic failure: allowed malware/phishing
    ("WARN_USER", False): -0.5,     # Slight friction on safe URL
    ("WARN_USER", True): +5.0,      # Helpful warning on suspicious URL
    ("DEEP_INSPECT", False): -0.2,  # Latency cost on safe URL
    ("DEEP_INSPECT", True): +3.0,   # Useful deep inspection
    ("BLOCK", False): -5.0,         # False positive: user blocked from safe QR
    ("BLOCK", True): +10.0          # Success: malicious attack neutralized
}


def is_supabase_configured() -> bool:
    """Check if Supabase credentials have been supplied via environment variables.
    
    Returns:
        bool: True if both SUPABASE_URL and SUPABASE_KEY are non-empty and non-placeholder.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    if "your-project-id" in SUPABASE_URL or "your-supabase" in SUPABASE_KEY:
        return False
    return True
