"""URL Analyzer Module for QRShield.

Validates and statically analyzes decoded QR payloads, normalizes URLs safely,
extracts structural components, and computes preliminary heuristic risk assessments.

Security Policy:
- Static analysis is strictly safe: parses payloads without remote script execution.
- Bounded network checks (redirect and TLS inspection) enforce strict SSRF prevention,
  timeouts, and never execute client-side scripts.
"""

from dataclasses import asdict, dataclass, field
import ipaddress
import logging
import re
import socket
import ssl
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import tldextract

import config
from modules.feature_extractor import FeatureExtractor, extract_url_features

logger = logging.getLogger("qrshield.url_analyzer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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
    "office.com",
    "forms.office.com",
    "typeform.com",
    "surveymonkey.com",
    "slack.com",
    "zoom.com",
    "mozilla.org",
    "linkedin.com",
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
    "q",
    "uri",
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
    "dropbox",
    "twitter",
    "linkedin",
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

# Cyrillic lookalikes mimicking Latin characters
CYRILLIC_LOOKALIKES = {
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u043e": "o",
    "\u0440": "p", "\u0455": "s", "\u0456": "i", "\u0458": "j",
    "\u0445": "x", "\u0443": "y", "\u0432": "b", "\u043d": "h",
    "\u0442": "t", "\u043c": "m", "\u043a": "k", "\u045d": "i",
    "\u0491": "g", "\u0501": "d", "\u051b": "q", "\u051d": "w",
}

# Greek lookalikes mimicking Latin characters
GREEK_LOOKALIKES = {
    "\u03b1": "a", "\u03bf": "o", "\u03bd": "v", "\u03c1": "p",
    "\u03c4": "t", "\u03b9": "i", "\u03ba": "k", "\u03b5": "e",
}

# Private / local IP ranges for SSRF prevention
PRIVATE_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def is_private_or_local_host(hostname: str) -> bool:
    """Check if hostname resolves to localhost or private/unroutable IP ranges."""
    if not hostname:
        return True
    clean = hostname.strip("[]").strip().lower()
    if clean in ("localhost", "local", "broadcasthost", "loopback"):
        return True
    try:
        ip = ipaddress.ip_address(clean)
        return any(ip in net for net in PRIVATE_IP_NETWORKS)
    except ValueError:
        return False


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


@dataclass
class IndicatorRecord:
    """Documented security indicator finding with explicit weight and severity."""

    indicator_id: str
    category: str      # 'URL_STRUCTURE', 'DOMAIN_IMPERSONATION', 'PHISHING', 'REPUTATION', 'REDIRECT', 'TLS', 'QR_PAYLOAD'
    severity: str      # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'
    weight: float      # Score contribution (0.0 to 100.0)
    description: str   # Human-readable explanation
    evidence_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "category": self.category,
            "severity": self.severity,
            "weight": self.weight,
            "description": self.description,
            "evidence_data": self.evidence_data,
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
    risk_category: str = "LOW"  # 'LOW', 'SUSPICIOUS', 'HIGH', 'CRITICAL'
    risk_score: float = 0.0     # 0.0 to 1.0 (normalized)
    risk_indicators: List[str] = field(default_factory=list)
    indicator_records: List[IndicatorRecord] = field(default_factory=list)

    @property
    def risk_score_100(self) -> float:
        """Risk score on the 0 to 100 scale."""
        return round(self.risk_score * 100.0, 1)

    @property
    def verdict(self) -> str:
        """Derive standard threat classification verdict: SAFE, SUSPICIOUS, UNVERIFIED, or MALICIOUS."""
        if (
            self.status in ("DANGEROUS_SCHEME", "MALFORMED_URL", "MISSING_HOSTNAME")
            or self.risk_category in ("HIGH", "CRITICAL")
            or self.risk_score >= 0.50
        ):
            return "MALICIOUS"
        if self.risk_category in ("SUSPICIOUS", "MEDIUM") or self.risk_score >= 0.20:
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
            "risk_score_100": self.risk_score_100,
            "risk_indicators": self.risk_indicators,
            "suspicious_indicators": self.risk_indicators,
            "indicator_records": [ir.to_dict() for ir in self.indicator_records],
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


def detect_homoglyphs(domain: str) -> List[IndicatorRecord]:
    """Detect Cyrillic, Greek, or Punycode IDN homoglyphs used for visual spoofing."""
    clean = (domain or "").lower().strip()
    if not clean:
        return []

    records: List[IndicatorRecord] = []

    # Check for Punycode encoding
    if "xn--" in clean:
        records.append(
            IndicatorRecord(
                indicator_id="IND_HOMOGLYPH_DOMAIN",
                category="DOMAIN_IMPERSONATION",
                severity="HIGH",
                weight=60.0,
                description="Internationalized Domain Name (IDN Punycode 'xn--') detected. Often used in visual lookalike spoofing.",
                evidence_data={"domain": clean, "type": "punycode"},
            )
        )
        return records

    # Scan for confusable characters
    found_chars = []
    for ch in clean:
        if ch in CYRILLIC_LOOKALIKES:
            found_chars.append(f"Cyrillic (U+{ord(ch):04X}) mimicking '{CYRILLIC_LOOKALIKES[ch]}'")
        elif ch in GREEK_LOOKALIKES:
            found_chars.append(f"Greek (U+{ord(ch):04X}) mimicking '{GREEK_LOOKALIKES[ch]}'")
        elif ord(ch) > 127:
            found_chars.append(f"Non-ASCII char (U+{ord(ch):04X})")

    if found_chars:
        details = ", ".join(found_chars[:3])
        records.append(
            IndicatorRecord(
                indicator_id="IND_HOMOGLYPH_DOMAIN",
                category="DOMAIN_IMPERSONATION",
                severity="HIGH",
                weight=60.0,
                description=f"Homoglyph character spoofing detected in hostname: {details}.",
                evidence_data={"domain": clean, "confusables": found_chars},
            )
        )

    return records


def detect_typosquatting(domain: str, trusted_roots: Optional[Set[str]] = None) -> List[IndicatorRecord]:
    """Detect brand typosquatting permutations using Levenshtein distance."""
    clean = (domain or "").lower().strip()
    if not clean:
        return []

    trusted = trusted_roots or TRUSTED_ROOT_DOMAINS
    # Never flag genuine trusted domains
    for t in trusted:
        if clean == t or clean.endswith(f".{t}"):
            return []

    records: List[IndicatorRecord] = []
    labels = clean.split(".")

    for brand in HIGH_PROFILE_BRANDS:
        # Check for brand token in domain
        if brand in clean:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_BRAND_IMPERSONATION",
                    category="DOMAIN_IMPERSONATION",
                    severity="HIGH",
                    weight=40.0,
                    description=f"Deceptive brand impersonation signature detected (domain contains '{brand}' but is not an official {brand} property).",
                    evidence_data={"brand": brand, "domain": clean},
                )
            )
            return records

    for label in labels:
        if len(label) < 4:
            continue
        for brand in HIGH_PROFILE_BRANDS:
            # Check edit distance for typosquatting
            dist = levenshtein_distance(label, brand)
            if 1 <= dist <= 2 and len(label) >= 4 and len(brand) >= 4:
                records.append(
                    IndicatorRecord(
                        indicator_id="IND_TYPOSQUATTING",
                        category="DOMAIN_IMPERSONATION",
                        severity="HIGH",
                        weight=45.0,
                        description=f"Typosquatting detected: Hostname label '{label}' is a deceptive permutation of '{brand}' (edit distance {dist}).",
                        evidence_data={"brand": brand, "distance": dist, "label": label, "domain": clean},
                    )
                )
                return records

    return records


def inspect_url_structure(raw_url: str, parsed: Any, hostname: str, raw_payload: Optional[str] = None) -> List[IndicatorRecord]:
    """Analyze URL syntax for deceptive authority parsing, backslashes, ports, and obfuscations."""
    records: List[IndicatorRecord] = []
    clean_url = (raw_url or "").strip()
    raw_text = (raw_payload or clean_url).strip()

    # 1. Embedded credentials or deceptive userinfo (@ symbol)
    try:
        raw_parsed = urlparse(raw_text)
    except Exception:
        raw_parsed = None

    has_at_netloc = False
    if raw_parsed:
        has_at_netloc = bool(raw_parsed.username or raw_parsed.password or ("@" in (raw_parsed.netloc or "")))
    if not has_at_netloc:
        has_at_netloc = "@" in raw_text.split("?")[0].split("#")[0]

    if has_at_netloc:
        records.append(
            IndicatorRecord(
                indicator_id="IND_DECEPTIVE_USERINFO",
                category="URL_STRUCTURE",
                severity="HIGH",
                weight=50.0,
                description="Deceptive authority structure: URL contains embedded credentials or '@' symbol obscuring true destination host.",
                evidence_data={"raw": raw_text},
            )
        )

    # 2. Deceptive backslash in authority / netloc
    if "\\" in (parsed.netloc or "") or ("\\" in clean_url.split("?")[0].split("#")[0][:60]):
        records.append(
            IndicatorRecord(
                indicator_id="IND_MALFORMED_SYNTAX",
                category="URL_STRUCTURE",
                severity="HIGH",
                weight=85.0,
                description="Malformed URL syntax: Deceptive backslash in authority or path exploitable for cross-parser destination confusion.",
                evidence_data={"url": clean_url},
            )
        )

    # 3. IP address hostname analysis
    from modules.feature_extractor import check_is_ip_address
    clean_host = (hostname or "").strip().lower()
    if clean_host and check_is_ip_address(clean_host):
        if is_private_or_local_host(clean_host):
            records.append(
                IndicatorRecord(
                    indicator_id="IND_PRIVATE_IP_HOST",
                    category="URL_STRUCTURE",
                    severity="HIGH",
                    weight=45.0,
                    description=f"Target host is a private/loopback IP address ({clean_host}), indicating potential SSRF or local exploit.",
                    evidence_data={"host": clean_host},
                )
            )
        else:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_IP_HOST",
                    category="URL_STRUCTURE",
                    severity="HIGH",
                    weight=40.0,
                    description=f"Destination host is a numerical IP address ({clean_host}) instead of a registered domain (evasion indicator).",
                    evidence_data={"host": clean_host},
                )
            )

    # 4. Unusual / non-standard ports
    port = parsed.port
    if port is not None and port not in (80, 443, 8080, 8443):
        high_risk_ports = {21, 22, 23, 25, 53, 137, 445, 1433, 3306, 3389, 4444, 6667}
        if port in high_risk_ports:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_UNUSUAL_PORT",
                    category="URL_STRUCTURE",
                    severity="HIGH",
                    weight=40.0,
                    description=f"Dangerous/trojan non-web port :{port} specified (associated with database, shell, or C2 protocols).",
                    evidence_data={"port": port},
                )
            )
        else:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_UNUSUAL_PORT",
                    category="URL_STRUCTURE",
                    severity="MEDIUM",
                    weight=25.0,
                    description=f"Non-standard web service port :{port} specified.",
                    evidence_data={"port": port},
                )
            )

    # 5. Suspicious percent-encoding in path / query
    raw_norm = clean_url.lower()
    deceptive_encodings = ["%2f", "%5c", "%40", "%2e%2e", "%00", "%25"]
    if any(enc in raw_norm for enc in deceptive_encodings):
        records.append(
            IndicatorRecord(
                indicator_id="IND_SUSPICIOUS_ENCODING",
                category="URL_STRUCTURE",
                severity="MEDIUM",
                weight=25.0,
                description="Obfuscated percent-encoding detected masking path separators, traversal tokens, or embedded credentials.",
                evidence_data={"url": clean_url},
            )
        )

    # 6. Malicious executable / script file extensions in path
    path_lower = (parsed.path or "").lower()
    dangerous_exts = (".exe", ".scr", ".bat", ".apk", ".vbs", ".iso", ".cmd", ".ps1", ".jar")
    for dext in dangerous_exts:
        if path_lower.endswith(dext) or f"{dext}?" in path_lower:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_MALICIOUS_FILE_EXT",
                    category="PHISHING",
                    severity="HIGH",
                    weight=45.0,
                    description=f"Direct download of executable/script file ({dext}) detected in destination path.",
                    evidence_data={"extension": dext, "path": parsed.path},
                )
            )
            break

    return records


def inspect_safe_redirects(
    url: str,
    max_hops: int = 5,
    timeout: float = 2.5,
) -> Tuple[List[IndicatorRecord], Dict[str, Any]]:
    """Safely follow HTTP redirect headers with strict SSRF protection and bounded execution.

    Never executes page scripts or downloads full file bodies.
    """
    clean_url = (url or "").strip()
    records: List[IndicatorRecord] = []
    chain: List[Dict[str, Any]] = []

    try:
        parsed = urlparse(clean_url)
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return records, {"error": "Invalid URL structure"}

    if scheme not in ("http", "https") or not hostname or is_private_or_local_host(hostname):
        return records, {"skipped": "Private host or non-web scheme"}

    import httpx

    current_url = clean_url
    initial_domain = hostname
    is_cross_domain = False
    is_protocol_downgrade = False

    try:
        with httpx.Client(follow_redirects=False, timeout=timeout) as client:
            for hop in range(max_hops):
                try:
                    p_curr = urlparse(current_url)
                    host_curr = (p_curr.hostname or "").lower()
                    if is_private_or_local_host(host_curr):
                        records.append(
                            IndicatorRecord(
                                indicator_id="IND_SSRF_REDIRECT",
                                category="REDIRECT",
                                severity="HIGH",
                                weight=85.0,
                                description=f"Redirect chain attempted redirection to private internal host '{host_curr}' (SSRF hazard).",
                                evidence_data={"redirect_host": host_curr, "hop": hop},
                            )
                        )
                        break

                    headers = {"User-Agent": "QRShield-Redirect-Validator/2.0"}
                    # Try HEAD first
                    resp = client.head(current_url, headers=headers)
                    if resp.status_code in (405, 403):
                        # Fallback to streaming GET with no body read
                        with client.stream("GET", current_url, headers=headers) as stream_resp:
                            resp_status = stream_resp.status_code
                            resp_headers = stream_resp.headers
                    else:
                        resp_status = resp.status_code
                        resp_headers = resp.headers

                    chain.append({"hop": hop + 1, "url": current_url, "status": resp_status})

                    # Check for redirect status
                    if resp_status in (301, 302, 303, 307, 308):
                        loc = resp_headers.get("location")
                        if not loc:
                            break
                        next_url = urljoin(current_url, loc)
                        p_next = urlparse(next_url)
                        next_host = (p_next.hostname or "").lower()
                        next_scheme = (p_next.scheme or "").lower()

                        # Check protocol downgrade
                        if p_curr.scheme == "https" and next_scheme == "http":
                            is_protocol_downgrade = True

                        # Check cross-domain redirect
                        if next_host and next_host != initial_domain and not next_host.endswith(f".{initial_domain}"):
                            is_cross_domain = True

                        current_url = next_url
                    else:
                        break

                except Exception as hop_err:
                    chain.append({"hop": hop + 1, "url": current_url, "error": str(hop_err)})
                    break

        # Process detected redirect anomalies
        if len(chain) >= 4:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_EXCESSIVE_REDIRECTS",
                    category="REDIRECT",
                    severity="LOW",
                    weight=20.0,
                    description=f"Excessive redirection hops ({len(chain)} hops) detected, typical of evasion chains.",
                    evidence_data={"hop_count": len(chain)},
                )
            )

        if is_cross_domain:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_CROSS_DOMAIN_REDIRECT",
                    category="REDIRECT",
                    severity="LOW",
                    weight=15.0,
                    description=f"Cross-domain redirect detected: Origin '{initial_domain}' redirects to external destination.",
                    evidence_data={"initial_domain": initial_domain, "final_url": current_url},
                )
            )

        if is_protocol_downgrade:
            records.append(
                IndicatorRecord(
                    indicator_id="IND_PROTOCOL_DOWNGRADE",
                    category="REDIRECT",
                    severity="MEDIUM",
                    weight=30.0,
                    description="Security downgrade: Destination redirects from HTTPS encrypted transport to unencrypted HTTP.",
                    evidence_data={"initial_url": clean_url, "final_url": current_url},
                )
            )

    except Exception as exc:
        logger.debug("Safe redirect inspection error: %s", exc)

    return records, {
        "initial_url": clean_url,
        "final_url": current_url,
        "hop_count": len(chain),
        "chain": chain,
        "is_cross_domain": is_cross_domain,
        "is_protocol_downgrade": is_protocol_downgrade,
    }


def inspect_tls_certificate(hostname: str, port: int = 443, timeout: float = 2.0) -> Tuple[List[IndicatorRecord], Dict[str, Any]]:
    """Probe TLS handshake safely and inspect certificate validity without trusting HTTPS blindly."""
    clean_host = (hostname or "").strip().lower()
    records: List[IndicatorRecord] = []

    if not clean_host or is_private_or_local_host(clean_host):
        return records, {"status": "SKIPPED_PRIVATE_HOST"}

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((clean_host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=clean_host) as ssock:
                cert = ssock.getpeercert()
                not_after_str = cert.get("notAfter", "")
                subject = cert.get("subject", ())
                issuer = cert.get("issuer", ())

                # Validate expiration
                if not_after_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        expiry_dt = parsedate_to_datetime(not_after_str)
                        from datetime import datetime, timezone
                        if expiry_dt < datetime.now(timezone.utc):
                            records.append(
                                IndicatorRecord(
                                    indicator_id="IND_CERT_EXPIRED",
                                    category="TLS",
                                    severity="HIGH",
                                    weight=45.0,
                                    description=f"Expired TLS/SSL certificate detected (expired on {not_after_str}).",
                                    evidence_data={"notAfter": not_after_str},
                                )
                            )
                            return records, {"status": "EXPIRED", "expiry": not_after_str}
                    except Exception:
                        pass

                return records, {
                    "status": "VALID",
                    "expiry": not_after_str,
                    "cipher": ssock.cipher(),
                    "version": ssock.version(),
                }

    except ssl.SSLCertVerificationError as cert_err:
        records.append(
            IndicatorRecord(
                indicator_id="IND_CERT_INVALID",
                category="TLS",
                severity="HIGH",
                weight=45.0,
                description=f"TLS certificate verification failed: {cert_err.verify_message} (possible self-signed or MITM attack).",
                evidence_data={"error": str(cert_err)},
            )
        )
        return records, {"status": "INVALID_CERTIFICATE", "error": str(cert_err)}

    except (socket.timeout, TimeoutError):
        return records, {"status": "TIMEOUT", "error": "TLS handshake probe timed out"}

    except Exception as exc:
        return records, {"status": "CONNECTION_FAILED", "error": str(exc)}


class URLAnalyzer:
    """Analyzes and validates URLs, extracting features and preliminary heuristic risk."""

    def __init__(
        self,
        suspicious_keywords: Optional[List[str]] = None,
        suspicious_tlds: Optional[List[str]] = None,
        low_risk_max: Optional[float] = None,
        med_risk_max: Optional[float] = None,
    ) -> None:
        self.feature_extractor = FeatureExtractor(suspicious_keywords=suspicious_keywords)
        self.suspicious_tlds = set(
            tld.lower().strip(".")
            for tld in (suspicious_tlds or getattr(config, "DEFAULT_SUSPICIOUS_TLDS", []))
        )
        self.low_risk_max = low_risk_max if low_risk_max is not None else getattr(config, "HEURISTIC_RISK_LOW_MAX", 20.0)
        self.med_risk_max = med_risk_max if med_risk_max is not None else getattr(config, "HEURISTIC_RISK_MED_MAX", 50.0)

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
        raw_payload: Optional[str] = None,
    ) -> Tuple[str, float, List[str], List[IndicatorRecord]]:
        """Calculate centralized deterministic risk score and collect deduplicated indicators.

        Args:
            features: 15 canonical features from FeatureExtractor.
            status: Validation status string.
            hostname: Hostname string.
            query_params: Parsed query parameter dictionary.
            normalized_url: Normalized URL string.
            raw_payload: Decoded raw payload.

        Returns:
            Tuple[str, float, List[str], List[IndicatorRecord]]:
            (risk_category, risk_score_0_to_1, indicator_descriptions, indicator_records)
        """
        # 1. Non-standard or dangerous statuses
        if status == "DANGEROUS_SCHEME":
            rec = IndicatorRecord(
                indicator_id="IND_DANGEROUS_SCHEME",
                category="QR_PAYLOAD",
                severity="CRITICAL",
                weight=100.0,
                description="Weaponized URI scheme detected (e.g. javascript:, data:, file:). High exploit hazard.",
            )
            return "HIGH", 1.0, [rec.description], [rec]

        if status == "UNSUPPORTED_SCHEME":
            rec = IndicatorRecord(
                indicator_id="IND_UNSUPPORTED_SCHEME",
                category="QR_PAYLOAD",
                severity="MEDIUM",
                weight=25.0,
                description="Specialized non-web URI scheme (e.g. tel:, sms:, wifi:). Requires manual caution.",
            )
            return "MEDIUM", 0.25, [rec.description], [rec]

        if status in ("MALFORMED_URL", "MISSING_HOSTNAME"):
            rec = IndicatorRecord(
                indicator_id="IND_MALFORMED_SYNTAX",
                category="URL_STRUCTURE",
                severity="HIGH",
                weight=85.0,
                description="Malformed URL structure with invalid or deceptive syntax.",
            )
            return "HIGH", 0.85, [rec.description], [rec]

        if status in ("PLAIN_TEXT", "EMPTY_PAYLOAD"):
            rec = IndicatorRecord(
                indicator_id="PAYLOAD_PLAIN_TEXT",
                category="QR_PAYLOAD",
                severity="INFO",
                weight=0.0,
                description="Plain text payload containing no active web destination.",
            )
            return "LOW", 0.0, [rec.description], [rec]

        # Use an indicator map keyed by indicator_id to PREVENT DUPLICATES
        indicator_map: Dict[str, IndicatorRecord] = {}

        clean_host = (hostname or "").lower().strip()
        target_url = normalized_url or raw_payload or ""

        # Parse candidate URL components for structural inspection
        try:
            parsed_url = urlparse(target_url)
        except Exception:
            parsed_url = None

        # Check domain reputation against trusted roots
        is_trusted = False
        if clean_host:
            for trusted in TRUSTED_ROOT_DOMAINS:
                if clean_host == trusted or clean_host.endswith(f".{trusted}"):
                    is_trusted = True
                    break

        # 2. Homoglyph and Punycode IDN check
        if clean_host:
            hg_records = detect_homoglyphs(clean_host)
            for r in hg_records:
                indicator_map[r.indicator_id] = r

        # 3. Typosquatting check against protected brands
        if clean_host and not is_trusted:
            ts_records = detect_typosquatting(clean_host)
            for r in ts_records:
                indicator_map[r.indicator_id] = r

        # 4. URL structure & syntax anomalies
        if parsed_url:
            struct_records = inspect_url_structure(target_url, parsed_url, clean_host, raw_payload=raw_payload)
            for r in struct_records:
                indicator_map[r.indicator_id] = r

        # 5. Misleading subdomains on untrusted hosts
        if clean_host and not is_trusted:
            subdomain_parts = clean_host.split(".")[:-2]  # parts before SLD.TLD
            for part in subdomain_parts:
                for brand in HIGH_PROFILE_BRANDS:
                    if part == brand or brand in part:
                        indicator_map["IND_MISLEADING_SUBDOMAIN"] = IndicatorRecord(
                            indicator_id="IND_MISLEADING_SUBDOMAIN",
                            category="DOMAIN_IMPERSONATION",
                            severity="MEDIUM",
                            weight=35.0,
                            description=f"Deceptive subdomain structure: Brand name '{brand}' embedded in subdomain on untrusted host '{clean_host}'.",
                            evidence_data={"brand": brand, "host": clean_host},
                        )
                        break

        # 6. URL shorteners
        if clean_host:
            for shortener in URL_SHORTENERS:
                if clean_host == shortener or clean_host.endswith(f".{shortener}"):
                    indicator_map["IND_URL_SHORTENER"] = IndicatorRecord(
                        indicator_id="IND_URL_SHORTENER",
                        category="REPUTATION",
                        severity="LOW",
                        weight=20.0,
                        description=f"Known URL shortener detected ({clean_host}); masks destination endpoint.",
                        evidence_data={"shortener": clean_host},
                    )
                    break

        # 7. Open redirect query parameters
        if query_params:
            for param, vals in query_params.items():
                if param.lower() in REDIRECT_PARAM_NAMES:
                    for v in vals:
                        if "://" in v or v.startswith("//"):
                            indicator_map["IND_OPEN_REDIRECT_PARAM"] = IndicatorRecord(
                                indicator_id="IND_OPEN_REDIRECT_PARAM",
                                category="PHISHING",
                                severity="MEDIUM",
                                weight=25.0,
                                description=f"Potential unvalidated open-redirect parameter detected ('{param}={v[:35]}...').",
                                evidence_data={"param": param, "target": v},
                            )
                            break

        # 8. Excessive hyphens in untrusted host
        num_hyphens = int(features.get("num_hyphens_hostname", 0))
        if not is_trusted and num_hyphens >= 3:
            indicator_map["IND_EXCESSIVE_HYPHENS"] = IndicatorRecord(
                indicator_id="IND_EXCESSIVE_HYPHENS",
                category="URL_STRUCTURE",
                severity="LOW",
                weight=15.0,
                description=f"Excessive hyphens in hostname ({num_hyphens}), typical in brand-spoofing setups.",
                evidence_data={"hyphens": num_hyphens},
            )

        # 9. Excessive subdomains in untrusted host
        num_sub = int(features.get("num_subdomains", 0))
        if not is_trusted and num_sub >= 3:
            indicator_map["IND_EXCESSIVE_SUBDOMAINS"] = IndicatorRecord(
                indicator_id="IND_EXCESSIVE_SUBDOMAINS",
                category="URL_STRUCTURE",
                severity="LOW",
                weight=15.0,
                description=f"Excessive subdomains ({num_sub}), common in dynamic phishing campaigns.",
                evidence_data={"subdomains": num_sub},
            )

        # 10. High dot count in hostname
        num_dots = int(features.get("num_dots_hostname", 0))
        if not is_trusted and num_dots >= 4:
            indicator_map["IND_HIGH_DOT_COUNT"] = IndicatorRecord(
                indicator_id="IND_HIGH_DOT_COUNT",
                category="URL_STRUCTURE",
                severity="LOW",
                weight=10.0,
                description=f"High dot count in hostname ({num_dots}).",
                evidence_data={"dots": num_dots},
            )

        # 11. Credential-harvesting keywords in untrusted domain
        num_kw = int(features.get("num_suspicious_keywords", 0))
        if num_kw > 0 and not is_trusted:
            indicator_map["IND_CREDENTIAL_KEYWORDS"] = IndicatorRecord(
                indicator_id="IND_CREDENTIAL_KEYWORDS",
                category="PHISHING",
                severity="LOW",
                weight=15.0,
                description=f"Detected {num_kw} high-risk credential-harvesting keyword(s) (e.g. login, verify, account, banking).",
                evidence_data={"count": num_kw},
            )

        # 12. High-risk / disposable scam TLD check
        if clean_host and not is_trusted:
            try:
                extracted = tldextract.extract(clean_host)
                tld = (extracted.suffix or "").lower()
                if tld in self.suspicious_tlds:
                    indicator_map["IND_SUSPICIOUS_TLD"] = IndicatorRecord(
                        indicator_id="IND_SUSPICIOUS_TLD",
                        category="REPUTATION",
                        severity="LOW",
                        weight=20.0,
                        description=f"High-risk Top-Level Domain (.{tld}) frequently used in disposable scam sites.",
                        evidence_data={"tld": tld},
                    )
            except Exception:
                pass

        # 13. Insecure cleartext HTTP vs HTTPS transport
        if features.get("is_https", 0) == 0:
            indicator_map["IND_INSECURE_HTTP"] = IndicatorRecord(
                indicator_id="IND_INSECURE_HTTP",
                category="TLS",
                severity="INFO",
                weight=5.0,
                description="Insecure HTTP protocol (unencrypted cleartext transport; vulnerable to eavesdropping and interception).",
            )
        else:
            indicator_map["IND_HTTPS_ENCRYPTED"] = IndicatorRecord(
                indicator_id="IND_HTTPS_ENCRYPTED",
                category="TLS",
                severity="INFO",
                weight=0.0,
                description="HTTPS transport encryption in use (note: SSL encryption does not guarantee domain legitimacy).",
            )

        # Calculate deterministic sum of deduplicated indicator weights
        score = sum(rec.weight for rec in indicator_map.values())
        final_score_100 = max(0.0, min(100.0, score))
        risk_score_norm = round(final_score_100 / 100.0, 3)

        # Canonical heuristic categories
        if final_score_100 >= self.med_risk_max:
            category = "HIGH"
        elif final_score_100 >= self.low_risk_max:
            category = "MEDIUM"
        else:
            category = "LOW"

        record_list = list(indicator_map.values())
        indicator_strings = [rec.description for rec in record_list]

        if not indicator_strings:
            clean_rec = IndicatorRecord(
                indicator_id="IND_BASELINE_CLEAN",
                category="REPUTATION",
                severity="INFO",
                weight=0.0,
                description="Standard domain structure; no heuristic red flags detected.",
            )
            indicator_strings.append(clean_rec.description)
            record_list.append(clean_rec)

        return category, risk_score_norm, indicator_strings, record_list

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
        risk_cat, risk_score, indicators, records = self.assess_heuristic_risk(
            features=features,
            status=status,
            hostname=hostname,
            query_params=query_dict,
            normalized_url=normalized,
            raw_payload=raw_payload,
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
            indicator_records=records,
        )


# Global singleton instance
_analyzer_instance = URLAnalyzer()


def analyze_url(raw_payload: str) -> URLAnalysisResult:
    """Convenience helper to analyze a URL using default URLAnalyzer."""
    return _analyzer_instance.analyze(raw_payload)
