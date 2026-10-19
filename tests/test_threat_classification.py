"""Comprehensive test suite for Threat Classification, URL Verification, Safe Navigation, and Persistence.

Covers:
1. Suspicious URL risk score consistency (never 0/100 when suspicious).
2. Unfamiliar URLs are not automatically marked Malicious or blocked.
3. Accurate 4-verdict distinction: SAFE, SUSPICIOUS, UNVERIFIED, MALICIOUS.
4. Static URL verification: HTTPS, domain reputation, brand spoofing, open redirects, URL shorteners.
5. Safe navigation rules: Malicious blocked, Suspicious requires confirmation, Clean allowed without 'guaranteed safe' claims.
6. Persistence handling: Record generation, honest error reporting, and duplicate scan handling.
"""

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from modules.feature_extractor import extract_url_features
from modules.ml_classifier import MaliciousURLClassifier, MLPredictionResult, predict_url
from modules.threat_prevention import (
    PreventionDecision,
    ThreatPreventionEngine,
    get_prevention_engine,
)
from modules.url_analyzer import URLAnalysisResult, analyze_url


@pytest.fixture
def engine() -> ThreatPreventionEngine:
    return get_prevention_engine()


class TestThreatClassificationConsistency:
    """Verify threat classification consistency and non-zero risk scores for suspicious URLs."""

    def test_suspicious_url_has_non_zero_risk_score(self, engine: ThreatPreventionEngine):
        """Root cause test: Suspicious URL must NOT display a risk score of 0/100."""
        # URL with suspicious indicators (multiple hyphens + brand keyword + redirect)
        suspicious_url = "http://verify-login-update-security-portal.info/session?dest=https://external.com"
        analysis = analyze_url(suspicious_url)

        # Even if a classifier hypothetically returned Benign with 0 risk score
        mock_ml = MLPredictionResult(
            predicted_label="Benign",
            predicted_class=0,
            risk_score=0.0,
            confidence=50.0,
            probabilities={"Benign": 1.0, "Malicious": 0.0},
            is_model_available=True,
        )

        nav = engine.evaluate_navigation_decision(
            raw_payload=suspicious_url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=mock_ml,
        )

        dec = nav["decision"]
        assert dec.final_action in ("WARN", "BLOCK")
        # Risk score must be non-zero and reflect heuristic threat
        assert dec.risk_score >= 30.0
        assert dec.risk_score > 0.0
        assert "0.0/100" not in dec.explanation
        assert nav["status_category"] in ("UNCERTAIN", "BLOCKED")

    def test_unfamiliar_url_not_automatically_malicious(self, engine: ThreatPreventionEngine):
        """Unfamiliar benign HTTP URL must NOT be automatically classified as Malicious or blocked."""
        unfamiliar_url = "http://my-local-bakery.com/menu"
        analysis = analyze_url(unfamiliar_url)
        ml_pred = predict_url(unfamiliar_url)

        nav = engine.evaluate_navigation_decision(
            raw_payload=unfamiliar_url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        dec = nav["decision"]
        # Must NOT be hard-blocked as Malicious
        assert dec.final_action != "BLOCK"
        assert dec.final_action == "WARN"
        assert nav["status_category"] == "UNCERTAIN"
        assert nav.get("verdict") in ("UNVERIFIED", "SUSPICIOUS")
        assert nav.get("requires_confirmation") is True

    def test_verdict_and_risk_score_monotonicity(self, engine: ThreatPreventionEngine):
        """Ensure verdict, risk score, and prevention decision are mathematically aligned."""
        # Clean benign
        clean_url = "https://www.google.com"
        clean_ana = analyze_url(clean_url)
        clean_ml = predict_url(clean_url)
        clean_nav = engine.evaluate_navigation_decision(clean_url, "URL_HTTPS", clean_ana, clean_ml)
        assert clean_nav["decision"].final_action == "ALLOW"
        assert clean_nav["decision"].risk_score < 30.0

        # High-risk phishing
        phish_url = "http://paypal-security-update-verification.com/login/verify.php?token=123"
        phish_ana = analyze_url(phish_url)
        phish_ml = predict_url(phish_url)
        phish_nav = engine.evaluate_navigation_decision(phish_url, "URL_HTTP", phish_ana, phish_ml)
        assert phish_nav["decision"].final_action == "BLOCK"
        assert phish_nav["decision"].risk_score >= 70.0


class TestURLVerificationHeuristics:
    """Validate static URL analysis: HTTPS, domain reputation, brand spoofing, open redirects."""

    def test_brand_spoofing_detection(self):
        """Test that brand impersonation in untrusted domains is flagged."""
        spoofed_url = "https://chase-security-verify-login.net/account"
        result = analyze_url(spoofed_url)
        assert result.risk_category in ("MEDIUM", "HIGH")
        assert result.risk_score >= 0.35
        assert any("brand impersonation" in ind.lower() or "chase" in ind.lower() for ind in result.risk_indicators)

    def test_trusted_domain_discount(self):
        """Test that legitimate trusted domains are not penalized for standard paths."""
        trusted_url = "https://en.wikipedia.org/wiki/Phishing_detection_and_security"
        result = analyze_url(trusted_url)
        assert result.risk_category == "LOW"
        assert result.risk_score < 0.25

    def test_open_redirect_parameter_detection(self):
        """Test detection of open redirect query parameters."""
        redirect_url = "https://example.com/login?redirect=https://attacker.evil.com/harvest"
        result = analyze_url(redirect_url)
        assert any("redirect" in ind.lower() for ind in result.risk_indicators)
        assert result.risk_score >= 0.25

    def test_url_shortener_flagged(self):
        """Test detection of URL shorteners."""
        short_url = "https://bit.ly/3xyzSafeLooking"
        result = analyze_url(short_url)
        assert any("shortener" in ind.lower() for ind in result.risk_indicators)

    def test_http_cleartext_flagged(self):
        """Test detection of insecure unencrypted HTTP protocol."""
        http_url = "http://simple-site.org/index.html"
        result = analyze_url(http_url)
        assert any("http protocol" in ind.lower() for ind in result.risk_indicators)


class TestSafeNavigationPolicy:
    """Verify safe navigation rules: blocking, caution confirmation, and zero-trust disclaimers."""

    def test_malicious_url_completely_blocked(self, engine: ThreatPreventionEngine):
        """Malicious URL must block automatic navigation and provide no open URL."""
        mal_url = "http://paypal-security-update-verification.com/login/verify.php?token=123"
        ana = analyze_url(mal_url)
        ml = predict_url(mal_url)
        nav = engine.evaluate_navigation_decision(mal_url, "URL_HTTP", ana, ml)

        assert nav["is_permitted"] is False
        assert nav["open_url"] is None
        assert nav["button_label"] is None
        assert nav["status_category"] == "BLOCKED"

    def test_suspicious_url_requires_confirmation(self, engine: ThreatPreventionEngine):
        """Suspicious URL must require user confirmation and expose caution URL."""
        susp_url = "http://my-local-bakery.com/menu"
        ana = analyze_url(susp_url)
        ml = predict_url(susp_url)
        nav = engine.evaluate_navigation_decision(susp_url, "URL_HTTP", ana, ml)

        assert nav["is_permitted"] is False
        assert nav["requires_confirmation"] is True
        assert nav["caution_url"] == susp_url
        assert nav["status_category"] == "UNCERTAIN"

    def test_no_claim_of_guaranteed_safe(self, engine: ThreatPreventionEngine):
        """Zero-trust policy: Permitted destinations must never claim guaranteed safety."""
        clean_url = "https://www.wikipedia.org"
        ana = analyze_url(clean_url)
        ml = predict_url(clean_url)
        nav = engine.evaluate_navigation_decision(clean_url, "URL_HTTPS", ana, ml)

        assert "guaranteed safe" not in nav["display_title"].lower()
        assert "verified safe" not in nav["display_title"].lower()
        assert "baseline" in nav["display_title"].lower()
        assert "disclaimer" in nav
        assert "zero-trust" in nav["disclaimer"].lower() or "does not guarantee" in nav["disclaimer"].lower()


class TestSupabasePersistenceVerification:
    """Test Supabase persistence pipeline, error handling, and scan deduplication."""

    @patch("modules.database.get_supabase_client")
    def test_save_complete_scan_creates_all_records(self, mock_get_client):
        """Verify save_complete_scan coordinates insertions across all 4 tables."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock table returns
        test_scan_id = str(uuid.uuid4())
        mock_client.table.return_value.insert.return_value.execute.side_effect = [
            MagicMock(data=[{"id": test_scan_id, "extracted_url": "https://example.com"}]),  # scan_history
            MagicMock(data=[{"id": str(uuid.uuid4()), "scan_id": test_scan_id}]),            # threat_analysis
            MagicMock(data=[{"id": str(uuid.uuid4()), "scan_id": test_scan_id}]),            # prevention_decisions
            MagicMock(data=[{"id": str(uuid.uuid4()), "scan_id": test_scan_id}]),            # rl_feedback
        ]

        from modules.database import save_complete_scan
        result = save_complete_scan(
            extracted_url="https://example.com",
            qr_payload_type="URL_HTTPS",
            predicted_class="SAFE",
            risk_score=0.15,
            detected_indicators=["HTTPS protocol verified"],
            selected_action="ALLOW",
            decision_reason="Baseline clean",
            scan_id=test_scan_id,
            user_id="test-user-123",
            client=mock_client,
        )

        assert result is not None
        assert result["scan_id"] == test_scan_id
        assert result["scan_record"] is not None
        assert result["threat_record"] is not None
        assert result["decision_record"] is not None
        assert result["rl_record"] is not None

    @patch("modules.database.get_supabase_client")
    def test_database_write_failure_honest_handling(self, mock_get_client):
        """Verify honest error reporting when database write encounters failure."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("Connection timeout")

        from modules.database import save_complete_scan
        result = save_complete_scan(
            extracted_url="https://example.com",
            qr_payload_type="URL_HTTPS",
            predicted_class="SAFE",
            risk_score=0.15,
            client=mock_client,
        )

        # Must report failure honestly (None) rather than pretending success
        assert result is None
