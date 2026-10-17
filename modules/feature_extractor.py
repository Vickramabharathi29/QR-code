"""Feature Extractor Module for QRShield.

Extracts lexical, structural, and behavioral features from URLs and payloads
to feed into the machine learning classifier, heuristic risk engine, and RL state encoder.

All feature extractions are strictly static and do not send network requests.
Outputs are formatted for immediate compatibility with Scikit-learn models.
"""

import ipaddress
import re
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import parse_qs, urlparse

import pandas as pd
import tldextract

import config

# Ordered canonical feature names for Scikit-learn compatibility
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "num_dots_hostname",
    "num_subdomains",
    "is_https",
    "is_ip_address",
    "has_at_symbol",
    "num_hyphens_hostname",
    "num_digits_hostname",
    "num_suspicious_keywords",
    "num_special_chars",
    "path_length",
    "num_query_params",
    "has_url_encoding",
    "has_unusual_scheme",
]

# Set of special characters commonly analyzed in URL threat detection
SPECIAL_CHARACTERS: Set[str] = {
    "@", "-", "_", "?", "=", "&", "%", "+", "~", "$", "!", "*", ",", ";", ":", "/"
}

# Regex for URL encoding pattern (e.g., %20, %2F, %3D)
URL_ENCODING_REGEX = re.compile(r"%[0-9a-fA-F]{2}")


def check_is_ip_address(hostname: str) -> bool:
    """Check if hostname is an IPv4 or IPv6 address without DNS lookups."""
    if not hostname:
        return False
    clean_host = hostname.strip("[]").strip()
    # If IPv4 with port passed directly (e.g. 192.168.1.1:8080)
    if ":" in clean_host and clean_host.count(":") == 1:
        clean_host = clean_host.split(":")[0]
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def count_subdomains(hostname: str) -> int:
    """Count the number of subdomains in a hostname."""
    if not hostname or check_is_ip_address(hostname):
        return 0

    try:
        # Use offline tldextract
        extracted = tldextract.extract(hostname)
        if not extracted.subdomain:
            return 0
        return len(extracted.subdomain.split("."))
    except Exception:
        # Fallback to dot count minus domain and TLD parts
        parts = [p for p in hostname.split(".") if p]
        return max(0, len(parts) - 2)


class FeatureExtractor:
    """Extracts machine learning feature vectors from URLs and redirection metadata."""

    def __init__(self, suspicious_keywords: Optional[List[str]] = None) -> None:
        self.suspicious_keywords = [
            k.lower()
            for k in (suspicious_keywords or getattr(config, "DEFAULT_SUSPICIOUS_KEYWORDS", []))
        ]

    def extract_features(self, url: str) -> Dict[str, Union[int, float]]:
        """Extract the 15 standard numerical features from a URL or raw payload.

        Args:
            url: The URL or decoded string to analyze.

        Returns:
            Dict[str, Union[int, float]]: Dictionary mapping each canonical feature name
            to its numerical value.
        """
        raw_text = (url or "").strip()
        lower_text = raw_text.lower()

        # Parse URL safely
        try:
            parsed = urlparse(raw_text)
            scheme = (parsed.scheme or "").lower()
            hostname = (parsed.hostname or "").lower()
            path = parsed.path or ""
            query = parsed.query or ""
        except Exception:
            scheme = ""
            hostname = ""
            path = ""
            query = ""

        # 1. url_length
        url_len = len(raw_text)

        # 2. hostname_length
        host_len = len(hostname)

        # 3. num_dots_hostname
        num_dots = hostname.count(".")

        # 4. num_subdomains
        num_sub = count_subdomains(hostname)

        # 5. is_https
        is_https_val = 1 if scheme == "https" else 0

        # 6. is_ip_address
        is_ip_val = 1 if check_is_ip_address(hostname) else 0

        # 7. has_at_symbol
        has_at_val = 1 if "@" in raw_text else 0

        # 8. num_hyphens_hostname
        num_hyphens = hostname.count("-")

        # 9. num_digits_hostname
        num_digits = sum(1 for c in hostname if c.isdigit())

        # 10. num_suspicious_keywords
        num_keywords = 0
        for kw in self.suspicious_keywords:
            if kw and kw in lower_text:
                num_keywords += 1

        # 11. num_special_chars
        num_special = sum(1 for c in raw_text if c in SPECIAL_CHARACTERS)

        # 12. path_length
        path_len = len(path)

        # 13. num_query_params
        if query:
            try:
                params = parse_qs(query, keep_blank_values=True)
                num_params = len(params)
            except Exception:
                num_params = len([p for p in query.split("&") if p])
        else:
            num_params = 0

        # 14. has_url_encoding
        has_encoding_val = 1 if bool(URL_ENCODING_REGEX.search(raw_text)) else 0

        # 15. has_unusual_scheme
        if not scheme:
            # Plain text or missing scheme
            has_unusual_scheme_val = 1
        elif scheme in ("http", "https"):
            has_unusual_scheme_val = 0
        else:
            has_unusual_scheme_val = 1

        return {
            "url_length": url_len,
            "hostname_length": host_len,
            "num_dots_hostname": num_dots,
            "num_subdomains": num_sub,
            "is_https": is_https_val,
            "is_ip_address": is_ip_val,
            "has_at_symbol": has_at_val,
            "num_hyphens_hostname": num_hyphens,
            "num_digits_hostname": num_digits,
            "num_suspicious_keywords": num_keywords,
            "num_special_chars": num_special,
            "path_length": path_len,
            "num_query_params": num_params,
            "has_url_encoding": has_encoding_val,
            "has_unusual_scheme": has_unusual_scheme_val,
        }

    def to_vector(self, feature_dict: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to an ordered numerical vector.

        Guarantees identical feature ordering for Scikit-learn model inference.

        Args:
            feature_dict: Dictionary returned by extract_features().

        Returns:
            List[float]: Canonical float vector.
        """
        return [float(feature_dict.get(name, 0.0)) for name in FEATURE_NAMES]

    def to_dataframe(self, feature_dict: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Convert one or multiple feature dictionaries into a pandas DataFrame.

        Args:
            feature_dict: Single dictionary or list of dictionaries.

        Returns:
            pd.DataFrame: DataFrame with FEATURE_NAMES as columns.
        """
        if isinstance(feature_dict, dict):
            data = [{name: feature_dict.get(name, 0.0) for name in FEATURE_NAMES}]
        else:
            data = [{name: item.get(name, 0.0) for name in FEATURE_NAMES} for item in feature_dict]
        return pd.DataFrame(data, columns=FEATURE_NAMES)


# Global singleton instance
_extractor_instance = FeatureExtractor()


def extract_url_features(url: str) -> Dict[str, Union[int, float]]:
    """Convenience helper to extract feature dictionary from a URL."""
    return _extractor_instance.extract_features(url)


def extract_feature_vector(url: str) -> List[float]:
    """Convenience helper to extract canonical feature vector from a URL."""
    features = _extractor_instance.extract_features(url)
    return _extractor_instance.to_vector(features)
