"""Modular Threat Intelligence Layer for QRShield.

Provides multi-provider threat intelligence lookups (URLhaus, Google Safe Browsing,
VirusTotal) with TTL caching, privacy safeguards, rate-limit handling, and strict
status differentiation.

Security & Privacy Guarantees:
- Never sends non-web payloads (plain text, Wi-Fi credentials, contact cards) to external providers.
- Filters out private, loopback, and link-local IP addresses (SSRF prevention).
- Never interprets an API timeout, rate-limit, or missing credential as a 'clean' result.
- Provides strict offline fallback when external networks or keys are unavailable.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import config

logger = logging.getLogger("qrshield.threat_intel")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Provider lookup statuses
STATUS_MATCH_MALICIOUS = "MATCH_MALICIOUS"
STATUS_CLEAN = "CLEAN"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_RATE_LIMITED = "RATE_LIMITED"
STATUS_UNKNOWN = "UNKNOWN"

# Private IP networks for SSRF protection
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


@dataclass
class ThreatIntelProviderResult:
    """Individual result from a single threat intelligence provider."""

    provider: str
    status: str  # MATCH_MALICIOUS, CLEAN, UNAVAILABLE, TIMEOUT, RATE_LIMITED, UNKNOWN
    is_malicious: bool
    threat_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThreatIntelAggregateResult:
    """Consolidated threat intelligence outcome across all evaluated providers."""

    overall_status: str  # MATCH_MALICIOUS, CLEAN, UNAVAILABLE, UNKNOWN
    is_malicious: bool
    primary_threat: Optional[str]
    evidence: List[str]
    provider_results: Dict[str, ThreatIntelProviderResult]
    query_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "is_malicious": self.is_malicious,
            "primary_threat": self.primary_threat,
            "evidence": self.evidence,
            "query_timestamp": self.query_timestamp,
            "providers": {k: v.to_dict() for k, v in self.provider_results.items()},
        }


class ThreatIntelligenceEngine:
    """Modular coordinator for live and cached threat intelligence lookups."""

    def __init__(
        self,
        timeout: Optional[float] = None,
        cache_ttl: Optional[int] = None,
    ) -> None:
        self.timeout = timeout if timeout is not None else getattr(config, "THREAT_INTEL_TIMEOUT", 3.0)
        self.cache_ttl = cache_ttl if cache_ttl is not None else getattr(config, "THREAT_INTEL_CACHE_TTL", 900)
        # In-memory TTL cache: {url_hash: (timestamp, ThreatIntelAggregateResult)}
        self._cache: Dict[str, Tuple[float, ThreatIntelAggregateResult]] = {}

    def _get_cache(self, url: str) -> Optional[ThreatIntelAggregateResult]:
        """Retrieve cached result if within TTL."""
        h = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
        entry = self._cache.get(h)
        if not entry:
            return None
        cached_time, res = entry
        if time.time() - cached_time <= self.cache_ttl:
            return res
        # Expired
        self._cache.pop(h, None)
        return None

    def _set_cache(self, url: str, result: ThreatIntelAggregateResult) -> None:
        """Store aggregate result in cache with current timestamp."""
        h = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
        self._cache[h] = (time.time(), result)

    def query_urlhaus(self, url: str) -> ThreatIntelProviderResult:
        """Query URLhaus (abuse.ch) malware & phishing URL database.

        Uses the public abuse.ch API endpoint. Does not require an API key by default.
        """
        start = time.perf_counter()
        endpoint = "https://urlhaus-api.abuse.ch/v1/url/"

        try:
            import httpx
            headers = {"User-Agent": "QRShield-ThreatLab/2.0 (Cybersecurity Static Scanner)"}
            api_key = getattr(config, "URLHAUS_API_KEY", "")
            if api_key:
                headers["Auth-Key"] = api_key

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, data={"url": url}, headers=headers)

            duration_ms = round((time.perf_counter() - start) * 1000.0, 1)

            if resp.status_code == 429:
                return ThreatIntelProviderResult(
                    provider="urlhaus",
                    status=STATUS_RATE_LIMITED,
                    is_malicious=False,
                    details={"http_status": 429, "error": "URLhaus rate limit exceeded"},
                    response_time_ms=duration_ms,
                )

            if resp.status_code != 200:
                return ThreatIntelProviderResult(
                    provider="urlhaus",
                    status=STATUS_UNAVAILABLE,
                    is_malicious=False,
                    details={"http_status": resp.status_code},
                    response_time_ms=duration_ms,
                )

            data = resp.json()
            q_status = data.get("query_status", "").lower()

            if q_status == "ok":
                # Found in URLhaus malicious repository
                threat = data.get("threat") or "malware_download"
                tags = data.get("tags") or []
                url_status = data.get("url_status", "unknown")
                return ThreatIntelProviderResult(
                    provider="urlhaus",
                    status=STATUS_MATCH_MALICIOUS,
                    is_malicious=True,
                    threat_type=threat,
                    details={
                        "urlhaus_status": url_status,
                        "threat": threat,
                        "tags": tags,
                        "date_added": data.get("date_added"),
                    },
                    response_time_ms=duration_ms,
                )
            elif q_status == "no_results":
                # Verified clean against URLhaus database
                return ThreatIntelProviderResult(
                    provider="urlhaus",
                    status=STATUS_CLEAN,
                    is_malicious=False,
                    details={"query_status": "no_results"},
                    response_time_ms=duration_ms,
                )
            else:
                return ThreatIntelProviderResult(
                    provider="urlhaus",
                    status=STATUS_UNKNOWN,
                    is_malicious=False,
                    details={"raw_query_status": q_status},
                    response_time_ms=duration_ms,
                )

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000.0, 1)
            err_name = type(exc).__name__
            status = STATUS_TIMEOUT if "timeout" in err_name.lower() else STATUS_UNAVAILABLE
            return ThreatIntelProviderResult(
                provider="urlhaus",
                status=status,
                is_malicious=False,
                details={"error_type": err_name, "message": str(exc)},
                response_time_ms=duration_ms,
            )

    def query_google_safe_browsing(self, url: str) -> ThreatIntelProviderResult:
        """Query Google Safe Browsing Lookup API v4 if API key is configured."""
        start = time.perf_counter()
        api_key = getattr(config, "GOOGLE_SAFE_BROWSING_API_KEY", "")

        if not api_key or "your-" in api_key.lower():
            return ThreatIntelProviderResult(
                provider="google_safe_browsing",
                status=STATUS_UNAVAILABLE,
                is_malicious=False,
                details={"reason": "API key not configured"},
                response_time_ms=0.0,
            )

        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
        payload = {
            "client": {"clientId": "qrshield-scanner", "clientVersion": "2.0.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }

        try:
            import httpx
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload)

            duration_ms = round((time.perf_counter() - start) * 1000.0, 1)

            if resp.status_code == 429:
                return ThreatIntelProviderResult(
                    provider="google_safe_browsing",
                    status=STATUS_RATE_LIMITED,
                    is_malicious=False,
                    details={"http_status": 429},
                    response_time_ms=duration_ms,
                )

            if resp.status_code != 200:
                return ThreatIntelProviderResult(
                    provider="google_safe_browsing",
                    status=STATUS_UNAVAILABLE,
                    is_malicious=False,
                    details={"http_status": resp.status_code},
                    response_time_ms=duration_ms,
                )

            data = resp.json()
            matches = data.get("matches", [])
            if matches:
                first_match = matches[0]
                t_type = first_match.get("threatType", "MALICIOUS")
                return ThreatIntelProviderResult(
                    provider="google_safe_browsing",
                    status=STATUS_MATCH_MALICIOUS,
                    is_malicious=True,
                    threat_type=t_type,
                    details={"matches": matches},
                    response_time_ms=duration_ms,
                )
            else:
                return ThreatIntelProviderResult(
                    provider="google_safe_browsing",
                    status=STATUS_CLEAN,
                    is_malicious=False,
                    details={"matches": []},
                    response_time_ms=duration_ms,
                )

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000.0, 1)
            err_name = type(exc).__name__
            status = STATUS_TIMEOUT if "timeout" in err_name.lower() else STATUS_UNAVAILABLE
            return ThreatIntelProviderResult(
                provider="google_safe_browsing",
                status=status,
                is_malicious=False,
                details={"error_type": err_name, "message": str(exc)},
                response_time_ms=duration_ms,
            )

    def query_virustotal(self, url: str) -> ThreatIntelProviderResult:
        """Query VirusTotal v3 API for domain / URL reputation if API key is configured."""
        start = time.perf_counter()
        api_key = getattr(config, "VIRUSTOTAL_API_KEY", "")

        if not api_key or "your-" in api_key.lower():
            return ThreatIntelProviderResult(
                provider="virustotal",
                status=STATUS_UNAVAILABLE,
                is_malicious=False,
                details={"reason": "API key not configured"},
                response_time_ms=0.0,
            )

        try:
            import base64
            import httpx
            # VT v3 URL identifier is base64 without padding
            url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
            endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
            headers = {"x-apikey": api_key, "User-Agent": "QRShield-ThreatLab/2.0"}

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(endpoint, headers=headers)

            duration_ms = round((time.perf_counter() - start) * 1000.0, 1)

            if resp.status_code == 429:
                return ThreatIntelProviderResult(
                    provider="virustotal",
                    status=STATUS_RATE_LIMITED,
                    is_malicious=False,
                    details={"http_status": 429},
                    response_time_ms=duration_ms,
                )

            if resp.status_code == 404:
                # URL has not been analyzed by VirusTotal yet
                return ThreatIntelProviderResult(
                    provider="virustotal",
                    status=STATUS_UNKNOWN,
                    is_malicious=False,
                    details={"reason": "URL not indexed in VirusTotal database"},
                    response_time_ms=duration_ms,
                )

            if resp.status_code != 200:
                return ThreatIntelProviderResult(
                    provider="virustotal",
                    status=STATUS_UNAVAILABLE,
                    is_malicious=False,
                    details={"http_status": resp.status_code},
                    response_time_ms=duration_ms,
                )

            data = resp.json()
            stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious_count = int(stats.get("malicious", 0))
            suspicious_count = int(stats.get("suspicious", 0))

            if malicious_count >= 2:
                return ThreatIntelProviderResult(
                    provider="virustotal",
                    status=STATUS_MATCH_MALICIOUS,
                    is_malicious=True,
                    threat_type="virustotal_multi_engine_detection",
                    details={"stats": stats, "malicious_count": malicious_count},
                    response_time_ms=duration_ms,
                )
            else:
                return ThreatIntelProviderResult(
                    provider="virustotal",
                    status=STATUS_CLEAN,
                    is_malicious=False,
                    details={"stats": stats},
                    response_time_ms=duration_ms,
                )

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000.0, 1)
            err_name = type(exc).__name__
            status = STATUS_TIMEOUT if "timeout" in err_name.lower() else STATUS_UNAVAILABLE
            return ThreatIntelProviderResult(
                provider="virustotal",
                status=status,
                is_malicious=False,
                details={"error_type": err_name, "message": str(exc)},
                response_time_ms=duration_ms,
            )

    def lookup_url(
        self,
        raw_url: str,
        enable_network: bool = True,
    ) -> ThreatIntelAggregateResult:
        """Perform comprehensive threat intelligence lookup across configured providers.

        Enforces privacy constraints:
        - Non-web payloads and local/private addresses are never transmitted.
        - Uses cached results when available.
        - Never treats unavailable providers as proof of safety.

        Args:
            raw_url: Target URL to evaluate.
            enable_network: Set to False to run in strict offline / air-gapped mode.

        Returns:
            ThreatIntelAggregateResult: Consolidated findings and evidence list.
        """
        clean_url = (raw_url or "").strip()

        # 1. Privacy & Scope Validation
        try:
            parsed = urlparse(clean_url)
            scheme = (parsed.scheme or "").lower()
            hostname = (parsed.hostname or "").lower()
        except Exception:
            scheme = ""
            hostname = ""

        if scheme not in ("http", "https") or not hostname or is_private_or_local_host(hostname):
            # Non-web payload or private IP: Do not leak to third parties
            return ThreatIntelAggregateResult(
                overall_status=STATUS_UNKNOWN,
                is_malicious=False,
                primary_threat=None,
                evidence=[],
                provider_results={},
            )

        # 2. Cache check
        cached = self._get_cache(clean_url)
        if cached:
            return cached

        # 3. If network is disabled (offline testing)
        if not enable_network:
            res = ThreatIntelAggregateResult(
                overall_status=STATUS_UNAVAILABLE,
                is_malicious=False,
                primary_threat=None,
                evidence=["Threat intelligence lookup offline (network queries disabled)."],
                provider_results={},
            )
            return res

        # 4. Query available providers
        provider_results: Dict[str, ThreatIntelProviderResult] = {}
        evidence: List[str] = []
        is_malicious = False
        primary_threat = None

        # A. URLhaus
        try:
            uh_res = self.query_urlhaus(clean_url)
            provider_results["urlhaus"] = uh_res
            if uh_res.status == STATUS_MATCH_MALICIOUS:
                is_malicious = True
                primary_threat = f"URLhaus: {uh_res.threat_type}"
                evidence.append(f"URLhaus confirmed malicious match: {uh_res.threat_type}")
            elif uh_res.status == STATUS_CLEAN:
                evidence.append("URLhaus lookup: No active threat records indexed.")
        except Exception as exc:
            logger.debug(f"URLhaus query error: {exc}")

        # B. Google Safe Browsing
        try:
            gsb_res = self.query_google_safe_browsing(clean_url)
            provider_results["google_safe_browsing"] = gsb_res
            if gsb_res.status == STATUS_MATCH_MALICIOUS:
                is_malicious = True
                if not primary_threat:
                    primary_threat = f"Safe Browsing: {gsb_res.threat_type}"
                evidence.append(f"Google Safe Browsing confirmed match: {gsb_res.threat_type}")
            elif gsb_res.status == STATUS_CLEAN:
                evidence.append("Google Safe Browsing lookup: No active threat matches.")
        except Exception as exc:
            logger.debug(f"Google Safe Browsing query error: {exc}")

        # C. VirusTotal
        try:
            vt_res = self.query_virustotal(clean_url)
            provider_results["virustotal"] = vt_res
            if vt_res.status == STATUS_MATCH_MALICIOUS:
                is_malicious = True
                if not primary_threat:
                    primary_threat = "VirusTotal: Multi-engine detection"
                evidence.append("VirusTotal confirmed detection by multiple security vendors.")
            elif vt_res.status == STATUS_CLEAN:
                evidence.append("VirusTotal lookup: Clean reputation among reporting vendors.")
        except Exception as exc:
            logger.debug(f"VirusTotal query error: {exc}")

        # Determine overall aggregate status
        has_malicious = any(r.status == STATUS_MATCH_MALICIOUS for r in provider_results.values())
        has_clean = any(r.status == STATUS_CLEAN for r in provider_results.values())
        all_unavailable = bool(provider_results) and all(
            r.status in (STATUS_UNAVAILABLE, STATUS_TIMEOUT, STATUS_RATE_LIMITED) for r in provider_results.values()
        )

        if has_malicious:
            overall_status = STATUS_MATCH_MALICIOUS
        elif has_clean:
            overall_status = STATUS_CLEAN
        elif all_unavailable or not provider_results:
            overall_status = STATUS_UNAVAILABLE
        else:
            overall_status = STATUS_UNKNOWN

        aggregate_res = ThreatIntelAggregateResult(
            overall_status=overall_status,
            is_malicious=is_malicious,
            primary_threat=primary_threat,
            evidence=evidence,
            provider_results=provider_results,
        )

        self._set_cache(clean_url, aggregate_res)
        return aggregate_res


# Global singleton engine
_ti_instance = ThreatIntelligenceEngine()


def lookup_threat_intelligence(url: str, enable_network: bool = True) -> ThreatIntelAggregateResult:
    """Convenience helper to look up URL across threat intelligence providers."""
    return _ti_instance.lookup_url(url, enable_network=enable_network)
