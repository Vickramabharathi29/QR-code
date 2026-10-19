"""URL Analyzer Module for QRShield.

Validates and statically analyzes decoded QR payloads, normalizes URLs safely,
extracts structural components, and computes preliminary heuristic risk assessments.

Security Policy:
All operations in this module are strictly static. It NEVER makes HTTP/HTTPS requests,
resolves DNS records, follows redirects, or executes embedded content.
"""

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse, urlunparse

import tldextract

import config
from modules.feature_extractor import FeatureExtractor, extract_url_features

# Dangerous URI schemes that must never be treated as valid web URLs
DANGEROUS_SCHEMES = {
    "javascript",
    "data",
    "file",
    "vbscript",
    "blob",
    "about",
    "chrome",
    "ms-msdt",
    "search-ms",
    "shell",
    "jar",
    "filesystem",
}

# Recognized specialized/mobile schemes
SPECIALIZED_SCHEMES = {
    "mailto",
    "tel",
    "smsto",
    "sms",
    "wifi",
    "geo",
    "market",
    "intent",
    "facetime",
    "skype",
    "ftp",
    "sftp",
    "ssh",
}

# Recognized high-reputation domains (avoids false-positive flags on legitimate services)
TRUSTED_ROOT_DOMAINS = {
    "google.com",
    "wikipedia.org",
    "github.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "paypal.com",
    "chase.com",
    "youtube.com",
    "cloudflare.com",
    "forms.gle",
    "docs.google.com",
    "drive.google.com",
    "gov",
    "edu",
    "mil",
}

# Parameter names commonly abused in unvalidated open-redirect schemes
REDIRECT_PARAM_NAMES = {
    "url",
    "redirect",
    "redirect_url",
    "dest",
    "destination",
    "target",
    "next",
    "r",
    "u",
    "link",
    "to",
    "redir",
    "return_to",
    "continue",
    "goto",
}

# High-profile brands frequently targeted by deceptive quishing and homoglyph attacks
HIGH_PROFILE_BRANDS = {
    "paypal",
    "chase",
    "apple",
    "google",
    "microsoft",
    "bankofamerica",
    "wellsfargo",
    "citibank",
    "netflix",
    "amazon",
    "facebook",
    "instagram",
    "binance",
    "coinbase",
}

# Known URL shortener services that mask final destination
URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rb.gy",
    "shorturl.at",
}


@dataclass
class URLAnalysisResult:
    """Detailed outcome of static URL analysis and heuristic threat evaluation."""

    raw_payload: str
    is_valid_url: bool
    status: str  # 'VALID_WEB_URL', 'DANGEROUS_SCHEME', 'UNSUPPORTED_SCHEME', 'PLAIN_TEXT', 'EMPTY_PAYLOAD', 'MALFORMED_URL', 'MISSING_HOSTNAME'
    validation_message: str
    normalized_url: Optional[str] = None
    scheme: Optional[str] = None
    hostname: Optional[str] = None
    path: Optional[str] = None
    query_params: Dict[str, List[str]] = field(default_factory=dict)
    features: Dict[str, Union[int, float]] = field(default_factory=dict)
    risk_category: str = "LOW"  # 'LOW', 'MEDIUM', 'HIGH'
    risk_score: float = 0.0  # 0.0 to 1.0
    risk_indicators: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Derive standard threat classification verdict: SAFE, SUSPICIOUS, UNVERIFIED, or MALICIOUS."""
        if self.status in ("DANGEROUS_SCHEME", "MALFORMED_URL", "MISSING_HOSTNAME") or self.risk_category == "HIGH" or self.risk_score >= 0.70:
            return "MALICIOUS"
        if self.risk_category == "MEDIUM" or self.risk_score >= 0.30:
            return "SUSPICIOUS"
        if self.scheme == "http" or self.status == "UNSUPPORTED_SCHEME":
            return "UNVERIFIED"
        return "SAFE"

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary format."""
        return {
            "raw_payload": self.raw_payload,
            "is_valid_url": self.is_valid_url,
            "status": self.status,
            "validation_message": self.validation_message,
            "normalized_url": self.normalized_url,
            "scheme": self.scheme,
            "hostname": self.hostname,
            "path": self.path,
            "features": self.features,
            "risk_category": self.risk_category,
            "risk_level": self.risk_category,
            "risk_score": self.risk_score,
            "risk_indicators": self.risk_indicators,
            "suspicious_indicators": self.risk_indicators,
            "verdict": self.verdict,
        }

    @property
    def suspicious_indicators(self) -> List[str]:
        """Backward-compatible alias for risk_indicators."""
        return self.risk_indicators

    @property
    def risk_level(self) -> str:
        """Backward-compatible alias for risk_category."""
        return self.risk_category


def normalize_url(raw_url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Safely normalize a URL without altering its semantic destination.

    Operations performed:
    - Strips leading and trailing whitespace
    - Lowercases protocol scheme and hostname
    - Removes default web ports (:80 for HTTP, :443 for HTTPS)
    - Preserves path and query parameters exactly as supplied

    Args:
        raw_url: The candidate URL string.

    Returns:
        Tuple[normalized_url, scheme, hostname, path, query] or (None, ...) if malformed.
    """
    clean = (raw_url or "").strip()
    if not clean:
        return None, None, None, None, None

    try:
        parsed = urlparse(clean)
        scheme = (parsed.scheme or "").lower()
        netloc = parsed.netloc or ""

        # Extract hostname and port
        hostname = (parsed.hostname or "").lower()
        port = parsed.port

        # Construct clean netloc without redundant default ports
        if port == 80 and scheme == "http":
            netloc = hostname
        elif port == 443 and scheme == "https":
            netloc = hostname
        elif hostname:
            netloc = f"{hostname}:{port}" if port else hostname

        # Reconstruct normalized URL
        normalized = urlunparse((
            scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))

        return normalized, scheme, hostname, parsed.path, parsed.query
    except Exception:
        return None, None, None, None, None


class URLAnalyzer:
    """Analyzes and validates URLs, extracting features and preliminary heuristic risk."""

    def __init__(
        self,
        suspicious_keywords: Optional[List[str]] = None,
        suspicious_tlds: Optional[List[str]] = None,
        low_risk_max: float = 30.0,
        med_risk_max: float = 60.0,
    ) -> None:
        self.feature_extractor = FeatureExtractor(suspicious_keywords=suspicious_keywords)
        self.suspicious_tlds = set(
            tld.lower().strip(".")
            for tld in (suspicious_tlds or getattr(config, "DEFAULT_SUSPICIOUS_TLDS", []))
        )
        self.low_risk_max = low_risk_max
        self.med_risk_max = med_risk_max

    def validate_payload(self, raw_payload: str) -> Tuple[bool, str, str, Optional[str], Optional[str], Optional[str]]:
        """Validate whether the payload is a valid web URL or another data type.

        Args:
            raw_payload: The decoded QR text.

        Returns:
            Tuple: (is_valid_url, status, message, normalized_url, scheme, hostname)
        """
        text = (raw_payload or "").strip()
        if not text:
            return (
                False,
                "EMPTY_PAYLOAD",
                "Payload is empty (0 characters).",
                None,
                None,
                None,
            )

        lower_text = text.lower()

        # Check for dangerous non-web schemes
        for d_scheme in DANGEROUS_SCHEMES:
            if lower_text.startswith(f"{d_scheme}:"):
                return (
                    False,
                    "DANGEROUS_SCHEME",
                    f"Dangerous URI scheme '{d_scheme}:' detected. This is not a safe web URL.",
                    None,
                    d_scheme,
                    None,
                )

        # Check for specialized/mobile schemes
        for s_scheme in SPECIALIZED_SCHEMES:
            if lower_text.startswith(f"{s_scheme}:"):
                return (
                    False,
                    "UNSUPPORTED_SCHEME",
                    f"Specialized scheme '{s_scheme}:' detected (not supported for standard web navigation).",
                    None,
                    s_scheme,
                    None,
                )

        # Parse candidate URL
        try:
            normalized, scheme, hostname, path, query = normalize_url(text)
        except Exception:
            return (
                False,
                "MALFORMED_URL",
                "Failed to parse payload: malformed URL structure.",
                None,
                None,
                None,
            )

        # Missing scheme or plain text
        if not scheme:
            # Check if domain-like without scheme (e.g. www.example.com)
            if lower_text.startswith("www.") and "." in text:
                # Add default https prefix for evaluation
                prefixed = f"https://{text}"
                norm_p, sch_p, host_p, _, _ = normalize_url(prefixed)
                if host_p:
                    return (
                        True,
                        "VALID_WEB_URL",
                        "Valid web URL (missing scheme; normalized to https).",
                        norm_p,
                        sch_p,
                        host_p,
                    )
            return (
                False,
                "PLAIN_TEXT",
                "Payload is plain text and does not contain a web URL scheme.",
                None,
                None,
                None,
            )

        # Only HTTP and HTTPS are valid web URLs
        if scheme not in ("http", "https"):
            return (
                False,
                "UNSUPPORTED_SCHEME",
                f"Unsupported protocol scheme '{scheme}:'. QRShield only validates HTTP and HTTPS web URLs.",
                None,
                scheme,
                None,
            )

        # Missing or invalid hostname
        if not hostname or hostname.startswith(".") or " " in hostname:
            return (
                False,
                "MISSING_HOSTNAME",
                "Malformed URL: missing or invalid hostname/domain.",
                None,
                scheme,
                None,
            )

        # Check for basic hostname structure
        if "." not in hostname and not self._is_valid_ip(hostname):
            return (
                False,
                "MALFORMED_URL",
                f"Invalid hostname '{hostname}': must be a valid domain or IP address.",
                None,
                scheme,
                hostname,
            )

        return (
            True,
            "VALID_WEB_URL",
            "Valid web URL format.",
            normalized,
            scheme,
            hostname,
        )

    def _is_valid_ip(self, host: str) -> bool:
        """Check if host is an IPv4 or IPv6 address."""
        from modules.feature_extractor import check_is_ip_address
        return check_is_ip_address(host)

    def assess_heuristic_risk(
        self,
        features: Dict[str, Union[int, float]],
        status: str,
        hostname: Optional[str] = None,
        query_params: Optional[Dict[str, List[str]]] = None,
        normalized_url: Optional[str] = None,
    ) -> Tuple[str, float, List[str]]:
        """Calculate preliminary heuristic risk score and collect indicator reasons.

        Notice: This is a static rule-based heuristic, NOT the trained machine learning model.

        Args:
            features: 15 canonical features from FeatureExtractor.
            status: Validation status string.
            hostname: Hostname string for TLD check.
            query_params: Optional parsed query parameter dictionary.
            normalized_url: Optional normalized URL string.

        Returns:
            Tuple[str, float, List[str]]: (risk_category, risk_score_0_to_1, indicators)
        """
        score = 0.0
        indicators: List[str] = []

        # 1. Non-standard or dangerous statuses
        if status == "DANGEROUS_SCHEME":
            return "HIGH", 1.0, [
                "Weaponized URI scheme detected (e.g. javascript:, data:, file:).",
                "High risk of arbitrary script execution, local file disclosure, or client exploitation.",
            ]

        if status == "UNSUPPORTED_SCHEME":
            return "HIGH", 0.70, [
                "Unusual or unsupported URI scheme.",
                "Non-web protocols can trigger unexpected external application handlers.",
            ]

        if status == "MALFORMED_URL" or status == "MISSING_HOSTNAME":
            return "HIGH", 0.65, [
                "Malformed URL structure with invalid or deceptive syntax.",
                "Syntax anomalies are frequently used to confuse standard security filters.",
            ]

        if status == "PLAIN_TEXT" or status == "EMPTY_PAYLOAD":
            return "LOW", 0.0, ["Plain text payload containing no active web destination."]

        # Check domain reputation against trusted roots
        clean_host = (hostname or "").lower().strip()
        is_trusted = False
        if clean_host:
            for trusted in TRUSTED_ROOT_DOMAINS:
                if clean_host == trusted or clean_host.endswith(f".{trusted}"):
                    is_trusted = True
                    break

        # 2. Hostname is an IP address
        if features.get("is_ip_address", 0) == 1:
            score += 35.0
            indicators.append("Host is an IP address instead of a domain name (evasion indicator).")

        # 3. Presence of '@' symbol
        if features.get("has_at_symbol", 0) == 1:
            score += 25.0
            indicators.append("URL contains '@' symbol (often used to obscure the destination host).")

        # 4. Brand spoofing / impersonation on untrusted hostnames
        if clean_host and not is_trusted:
            for brand in HIGH_PROFILE_BRANDS:
                if brand in clean_host:
                    score += 35.0
                    indicators.append(
                        f"Deceptive brand impersonation signature detected (domain contains '{brand}' but is not an official {brand} property)."
                    )
                    break

        # 5. URL Shorteners
        if clean_host:
            for shortener in URL_SHORTENERS:
                if clean_host == shortener or clean_host.endswith(f".{shortener}"):
                    score += 25.0
                    indicators.append(f"Known URL shortener detected ({clean_host}); masks ultimate destination endpoint.")
                    break

        # 6. Open-redirect query parameters
        if query_params:
            for param, vals in query_params.items():
                if param.lower() in REDIRECT_PARAM_NAMES:
                    for v in vals:
                        if "://" in v or v.startswith("//"):
                            score += 25.0
                            indicators.append(f"Potential unvalidated open-redirect parameter detected ('{param}={v[:35]}...').")
                            break

        # 7. Excessive hyphens in hostname (for non-trusted domains)
        num_hyphens = int(features.get("num_hyphens_hostname", 0))
        if not is_trusted:
            if num_hyphens >= 3:
                score += 20.0
                indicators.append(f"Excessive hyphens in hostname ({num_hyphens}), typical in brand-spoofing domains.")
            elif num_hyphens == 2:
                score += 10.0
                indicators.append(f"Multiple hyphens in hostname ({num_hyphens}).")

        # 8. Excessive subdomains (for non-trusted domains)
        num_sub = int(features.get("num_subdomains", 0))
        if not is_trusted:
            if num_sub >= 3:
                score += 20.0
                indicators.append(f"Excessive subdomains ({num_sub}), common in dynamic phishing setups.")
            elif num_sub == 2:
                score += 5.0
                indicators.append(f"Multiple subdomain levels ({num_sub}).")

        # 9. High dot count in hostname
        num_dots = int(features.get("num_dots_hostname", 0))
        if not is_trusted:
            if num_dots >= 4:
                score += 15.0
                indicators.append(f"High dot count in hostname ({num_dots}).")

        # 10. Suspicious keywords in URL
        num_kw = int(features.get("num_suspicious_keywords", 0))
        if num_kw > 0 and not is_trusted:
            kw_points = min(30.0, num_kw * 10.0)
            score += kw_points
            indicators.append(f"Detected {num_kw} high-risk keyword(s) (e.g. login, verify, account, banking).")

        # 11. High-risk TLD check
        if clean_host and not is_trusted:
            try:
                extracted = tldextract.extract(clean_host)
                tld = (extracted.suffix or "").lower()
                if tld in self.suspicious_tlds:
                    score += 20.0
                    indicators.append(f"High-risk Top-Level Domain (.{tld}) frequently used in disposable scam sites.")
            except Exception:
                pass

        # 12. Hexadecimal percent-encoding
        if features.get("has_url_encoding", 0) == 1:
            score += 10.0
            indicators.append("URL-encoding (%xx) detected, potentially masking redirected endpoints.")

        # 13. Excessive length
        url_len = int(features.get("url_length", 0))
        if url_len > 120:
            score += 15.0
            indicators.append(f"Abnormally long URL ({url_len} characters).")
        elif url_len > 80:
            score += 5.0
            indicators.append(f"Elevated URL length ({url_len} characters).")

        # 14. Insecure HTTP cleartext protocol
        if features.get("is_https", 0) == 0:
            score += 15.0
            indicators.append("Insecure HTTP protocol (unencrypted cleartext connection; vulnerable to interception).")
        else:
            indicators.append("HTTPS transport encryption in use (note: SSL encryption does not guarantee domain legitimacy).")

        # Reputable domain discount: cap baseline score for trusted platforms
        if is_trusted and score > 20.0:
            score = 15.0

        # Clamp score to [0.0, 100.0] and scale to [0.0, 1.0]
        final_score_100 = max(0.0, min(100.0, score))
        risk_score_norm = round(final_score_100 / 100.0, 3)

        if final_score_100 >= self.med_risk_max:
            category = "HIGH"
        elif final_score_100 >= self.low_risk_max:
            category = "MEDIUM"
        else:
            category = "LOW"

        if not indicators:
            indicators.append("Standard domain structure; no heuristic red flags detected.")

        return category, risk_score_norm, indicators

    def analyze(self, raw_payload: str) -> URLAnalysisResult:
        """Run complete static analysis on a decoded QR payload.

        Args:
            raw_payload: The raw decoded text from QR code.

        Returns:
            URLAnalysisResult: Full structured validation, features, and risk analysis.
        """
        is_valid, status, msg, normalized, scheme, hostname = self.validate_payload(raw_payload)

        # Extract 15 canonical features
        target_str = normalized if (is_valid and normalized) else raw_payload
        features = self.feature_extractor.extract_features(target_str)

        # Query parameters if valid URL
        query_dict: Dict[str, List[str]] = {}
        path_str: Optional[str] = None
        if normalized:
            try:
                parsed = urlparse(normalized)
                path_str = parsed.path
                if parsed.query:
                    query_dict = parse_qs(parsed.query, keep_blank_values=True)
            except Exception:
                pass

        # Heuristic risk assessment
        risk_cat, risk_score, indicators = self.assess_heuristic_risk(
            features=features,
            status=status,
            hostname=hostname,
            query_params=query_dict,
            normalized_url=normalized,
        )

        return URLAnalysisResult(
            raw_payload=raw_payload,
            is_valid_url=is_valid,
            status=status,
            validation_message=msg,
            normalized_url=normalized,
            scheme=scheme,
            hostname=hostname,
            path=path_str,
            query_params=query_dict,
            features=features,
            risk_category=risk_cat,
            risk_score=risk_score,
            risk_indicators=indicators,
        )


# Global singleton instance
_analyzer_instance = URLAnalyzer()


def analyze_url(raw_payload: str) -> URLAnalysisResult:
    """Convenience helper to analyze a URL using default URLAnalyzer."""
    return _analyzer_instance.analyze(raw_payload)
