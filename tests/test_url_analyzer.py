"""Unit tests for modules/url_analyzer.py.

Validates payload types, URL structure, safe normalization, and preliminary
heuristic risk scoring across benign, suspicious, malformed, IP-based, and
unsupported schemes using local test cases only.
"""

import pytest

from modules.url_analyzer import URLAnalyzer, analyze_url, normalize_url


# ---------------------------------------------------------------------------
# Test 1: Benign HTTPS URL
# ---------------------------------------------------------------------------
def test_analyze_benign_https_url():
    """Verify that legitimate HTTPS URLs are marked valid with LOW risk."""
    url = "https://www.google.com/search?q=cybersecurity"
    result = analyze_url(url)

    assert result.is_valid_url is True
    assert result.status == "VALID_WEB_URL"
    assert result.scheme == "https"
    assert result.hostname == "www.google.com"
    assert result.risk_category == "LOW"
    assert result.risk_score < 0.30
    assert len(result.features) == 15


# ---------------------------------------------------------------------------
# Test 2: Suspicious IP-Address URL
# ---------------------------------------------------------------------------
def test_analyze_ip_address_url():
    """Verify that IP-address URLs trigger elevated risk warnings."""
    url = "http://192.168.1.100:8080/admin/login?user=admin"
    result = analyze_url(url)

    assert result.is_valid_url is True
    assert result.status == "VALID_WEB_URL"
    assert result.features["is_ip_address"] == 1
    assert result.risk_category in ("MEDIUM", "HIGH")
    assert result.risk_score >= 0.30
    assert any("IP address" in ind for ind in result.risk_indicators)


# ---------------------------------------------------------------------------
# Test 3: Phishing / Deceptive Domain with Keywords & @ Symbol
# ---------------------------------------------------------------------------
def test_analyze_phishing_deceptive_url():
    """Verify that multiple phishing signals elevate risk to HIGH."""
    url = "http://paypal-security-update.account-verify.tk/login?user=target@bank.com"
    result = analyze_url(url)

    assert result.is_valid_url is True
    assert result.risk_category == "HIGH"
    assert result.risk_score >= 0.60
    assert len(result.risk_indicators) >= 3


# ---------------------------------------------------------------------------
# Test 4: Malformed URLs
# ---------------------------------------------------------------------------
def test_analyze_malformed_urls():
    """Verify that syntactically broken URLs are rejected safely."""
    # Missing host after scheme
    res_empty_host = analyze_url("http://")
    assert res_empty_host.is_valid_url is False
    assert res_empty_host.status in ("MISSING_HOSTNAME", "MALFORMED_URL")

    # Hostname with invalid characters/spaces
    res_space = analyze_url("http://bad domain.com/login")
    assert res_space.is_valid_url is False
    assert res_space.status in ("MISSING_HOSTNAME", "MALFORMED_URL")


# ---------------------------------------------------------------------------
# Test 5: Plain Text QR Payloads
# ---------------------------------------------------------------------------
def test_analyze_plain_text():
    """Verify that plain text payloads are categorized as non-URL text."""
    text = "Room 302 Projector Passcode: 4892"
    result = analyze_url(text)

    assert result.is_valid_url is False
    assert result.status == "PLAIN_TEXT"
    assert result.risk_category == "LOW"
    assert result.risk_score == 0.0


# ---------------------------------------------------------------------------
# Test 6: Empty or Whitespace Payloads
# ---------------------------------------------------------------------------
def test_analyze_empty_payload():
    """Verify handling of empty or blank strings."""
    result = analyze_url("   ")
    assert result.is_valid_url is False
    assert result.status == "EMPTY_PAYLOAD"


# ---------------------------------------------------------------------------
# Test 7: Dangerous URI Schemes (javascript, data, file)
# ---------------------------------------------------------------------------
def test_analyze_dangerous_schemes():
    """Verify that dangerous URI schemes are rejected and flagged as HIGH risk."""
    js_url = "javascript:alert(document.cookie)"
    res_js = analyze_url(js_url)
    assert res_js.is_valid_url is False
    assert res_js.status == "DANGEROUS_SCHEME"
    assert res_js.risk_category == "HIGH"
    assert res_js.risk_score == 1.0

    file_url = "file:///etc/passwd"
    res_file = analyze_url(file_url)
    assert res_file.is_valid_url is False
    assert res_file.status == "DANGEROUS_SCHEME"
    assert res_file.risk_category == "HIGH"

    data_url = "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=="
    res_data = analyze_url(data_url)
    assert res_data.is_valid_url is False
    assert res_data.status == "DANGEROUS_SCHEME"


# ---------------------------------------------------------------------------
# Test 8: Specialized / Mobile Schemes (tel, mailto, wifi)
# ---------------------------------------------------------------------------
def test_analyze_specialized_schemes():
    """Verify that non-web mobile schemes are flagged as UNSUPPORTED_SCHEME."""
    tel_payload = "tel:+18005550199"
    res_tel = analyze_url(tel_payload)
    assert res_tel.is_valid_url is False
    assert res_tel.status == "UNSUPPORTED_SCHEME"

    wifi_payload = "WIFI:S:GuestNet;T:WPA;P:password;;"
    res_wifi = analyze_url(wifi_payload)
    assert res_wifi.is_valid_url is False
    assert res_wifi.status == "UNSUPPORTED_SCHEME"


# ---------------------------------------------------------------------------
# Test 9: Safe Normalization Function
# ---------------------------------------------------------------------------
def test_normalize_url():
    """Verify that normalize_url removes default ports and lowercases host."""
    raw = "HTTPS://Example.COM:443/Path/Login?ref=123"
    norm, scheme, host, path, query = normalize_url(raw)

    assert norm == "https://example.com/Path/Login?ref=123"
    assert scheme == "https"
    assert host == "example.com"
    assert path == "/Path/Login"
    assert query == "ref=123"

    raw_http = "http://MY-SITE.ORG:80/home"
    norm_http, _, host_http, _, _ = normalize_url(raw_http)
    assert norm_http == "http://my-site.org/home"
    assert host_http == "my-site.org"


# ---------------------------------------------------------------------------
# Test 10: Serialization to Dictionary
# ---------------------------------------------------------------------------
def test_to_dict():
    """Verify that URLAnalysisResult converts cleanly to a dictionary."""
    result = analyze_url("https://github.com/torvalds/linux")
    d = result.to_dict()

    assert isinstance(d, dict)
    assert d["raw_payload"] == "https://github.com/torvalds/linux"
    assert d["is_valid_url"] is True
    assert "risk_category" in d
    assert "features" in d


# ---------------------------------------------------------------------------
# Test 11: Backward Compatibility Properties
# ---------------------------------------------------------------------------
def test_url_analysis_result_indicator_aliases():
    """Verify that URLAnalysisResult exposes both risk_indicators and suspicious_indicators."""
    result = analyze_url("https://www.google.com")
    assert hasattr(result, "risk_indicators")
    assert hasattr(result, "suspicious_indicators")
    assert isinstance(result.suspicious_indicators, list)
    assert result.suspicious_indicators == result.risk_indicators
    assert hasattr(result, "risk_level")
    assert result.risk_level == result.risk_category

    d = result.to_dict()
    assert "suspicious_indicators" in d
    assert "risk_indicators" in d
    assert "risk_level" in d
