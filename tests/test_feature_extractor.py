"""Unit tests for modules/feature_extractor.py.

Verifies accurate static extraction of the 15 canonical lexical and structural URL features
and conversion to Scikit-learn vectors and pandas DataFrames using local test cases only.
"""

import pandas as pd
import pytest

from modules.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
    extract_feature_vector,
    extract_url_features,
)


# ---------------------------------------------------------------------------
# Test 1: Canonical Feature List Completeness
# ---------------------------------------------------------------------------
def test_feature_names_list():
    """Verify that exactly 15 canonical features are defined."""
    assert len(FEATURE_NAMES) == 15
    assert "url_length" in FEATURE_NAMES
    assert "is_ip_address" in FEATURE_NAMES
    assert "has_at_symbol" in FEATURE_NAMES
    assert "num_suspicious_keywords" in FEATURE_NAMES
    assert "has_url_encoding" in FEATURE_NAMES


# ---------------------------------------------------------------------------
# Test 2: Standard Benign HTTPS URL Features
# ---------------------------------------------------------------------------
def test_extract_features_benign_https():
    """Verify feature values for a standard benign HTTPS URL."""
    url = "https://www.wikipedia.org/wiki/QR_code"
    features = extract_url_features(url)

    assert len(features) == 15
    assert features["url_length"] == len(url)
    assert features["hostname_length"] == len("www.wikipedia.org")
    assert features["num_dots_hostname"] == 2
    assert features["is_https"] == 1
    assert features["is_ip_address"] == 0
    assert features["has_at_symbol"] == 0
    assert features["num_hyphens_hostname"] == 0
    assert features["num_digits_hostname"] == 0
    assert features["has_unusual_scheme"] == 0
    assert features["path_length"] == len("/wiki/QR_code")
    assert features["num_query_params"] == 0


# ---------------------------------------------------------------------------
# Test 3: IP Address Hostname Detection
# ---------------------------------------------------------------------------
def test_extract_features_ip_address():
    """Verify detection of IPv4 and IPv6 hostnames."""
    ipv4_url = "http://192.168.1.50:8080/admin"
    f_v4 = extract_url_features(ipv4_url)
    assert f_v4["is_ip_address"] == 1
    assert f_v4["is_https"] == 0
    assert f_v4["num_digits_hostname"] > 0

    ipv6_url = "http://[2001:db8::1]/index.html"
    f_v6 = extract_url_features(ipv6_url)
    assert f_v6["is_ip_address"] == 1


# ---------------------------------------------------------------------------
# Test 4: Suspicious Keywords & @ Symbol
# ---------------------------------------------------------------------------
def test_extract_features_suspicious_phishing():
    """Verify feature extraction on a phishing-style deceptive URL."""
    url = "http://paypal-security-update.account-verify.tk/login?user=victim@bank.com"
    features = extract_url_features(url)

    assert features["has_at_symbol"] == 1
    assert features["num_hyphens_hostname"] >= 2
    assert features["num_suspicious_keywords"] >= 3  # paypal, account, verify, login
    assert features["is_https"] == 0
    assert features["num_query_params"] == 1


# ---------------------------------------------------------------------------
# Test 5: Subdomain Counting
# ---------------------------------------------------------------------------
def test_subdomain_counting():
    """Verify calculation of subdomain depth."""
    url = "https://login.portal.services.bank.example.com/auth"
    features = extract_url_features(url)
    # subdomains: login, portal, services, bank (4 subdomains)
    assert features["num_subdomains"] >= 3


# ---------------------------------------------------------------------------
# Test 6: URL Encoding Pattern Detection
# ---------------------------------------------------------------------------
def test_url_encoding_detection():
    """Verify detection of percent-encoded characters (%20, %2F)."""
    encoded_url = "https://example.com/search?q=hello%20world%2Ftest"
    f_encoded = extract_url_features(encoded_url)
    assert f_encoded["has_url_encoding"] == 1

    clean_url = "https://example.com/search?q=normal"
    f_clean = extract_url_features(clean_url)
    assert f_clean["has_url_encoding"] == 0


# ---------------------------------------------------------------------------
# Test 7: Dangerous and Unusual Schemes
# ---------------------------------------------------------------------------
def test_unusual_and_dangerous_schemes():
    """Verify detection of non-HTTP schemes."""
    js_payload = "javascript:alert(document.cookie)"
    f_js = extract_url_features(js_payload)
    assert f_js["has_unusual_scheme"] == 1
    assert f_js["is_https"] == 0

    plain_text = "Conference Room WiFi Pass: 12345"
    f_text = extract_url_features(plain_text)
    assert f_text["has_unusual_scheme"] == 1


# ---------------------------------------------------------------------------
# Test 8: Vector Conversion for Scikit-learn
# ---------------------------------------------------------------------------
def test_vector_conversion():
    """Verify that extract_feature_vector returns a numerical float list."""
    url = "https://secure.chase.com/login"
    vector = extract_feature_vector(url)

    assert isinstance(vector, list)
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vector)


# ---------------------------------------------------------------------------
# Test 9: DataFrame Conversion for Scikit-learn
# ---------------------------------------------------------------------------
def test_dataframe_conversion():
    """Verify conversion of feature dictionaries to pandas DataFrame."""
    extractor = FeatureExtractor()
    urls = [
        "https://www.example.com",
        "http://10.0.0.1/login",
    ]
    features_list = [extractor.extract_features(u) for u in urls]
    df = extractor.to_dataframe(features_list)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, len(FEATURE_NAMES))
    assert list(df.columns) == FEATURE_NAMES
    assert df["is_https"].iloc[0] == 1.0
    assert df["is_ip_address"].iloc[1] == 1.0
