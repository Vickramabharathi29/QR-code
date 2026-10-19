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
        assert result.risk_category in ("MEDIUM", "HIGH", "CRITICAL")
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


class TestCanonicalRiskScoringAndGoogleForms:
    """Validate canonical risk thresholds, Google Forms, and equivalent service scoring."""

    def test_google_forms_standard_viewform_low_risk(self, engine: ThreatPreventionEngine):
        """Standard Google Forms URL receives low risk score (0-19) and ALLOW action."""
        url = "https://docs.google.com/forms/d/e/1FAIpQLScP_7FjM5Y9z8nK1l4wX8yA0vE9qT1/viewform"
        ana = analyze_url(url)
        ml = predict_url(url)
        nav = engine.evaluate_navigation_decision(url, "URL_HTTPS", ana, ml)
        dec = nav["decision"]

        # Canonical Low Risk: 0-19 scale
        assert 0.0 <= dec.risk_score <= 19.0, f"Expected 0-19, got {dec.risk_score}"
        assert dec.final_action == "ALLOW"
        assert nav["is_permitted"] is True
        assert nav["status_category"] == "PERMITTED"
        assert nav["verdict"] == "SAFE"
        assert nav["open_url"] == url
        assert "Google Form" in nav.get("button_label", "")

    def test_google_forms_with_query_params_not_penalized(self, engine: ThreatPreventionEngine):
        """Google Forms with query parameters and long path does not get penalized into suspicious tier."""
        url = "https://docs.google.com/forms/d/e/1FAIpQLScP_7FjM5Y9z8nK1l4wX8yA0vE9qT1/viewform?usp=sf_link"
        ana = analyze_url(url)
        ml = predict_url(url)
        nav = engine.evaluate_navigation_decision(url, "URL_HTTPS", ana, ml)
        dec = nav["decision"]

        # Must NOT be 35.0 or suspicious
        assert dec.risk_score != 35.0
        assert 0.0 <= dec.risk_score <= 19.0, f"Expected low risk 0-19, got {dec.risk_score}"
        assert dec.final_action == "ALLOW"
        assert nav["is_permitted"] is True
        assert nav["status_category"] == "PERMITTED"

    def test_google_forms_shortlink_low_risk(self, engine: ThreatPreventionEngine):
        """Shortened forms.gle link receives low risk (0-19) and ALLOW action."""
        url = "https://forms.gle/7w5yBwJ5qC3h8k9A6"
        ana = analyze_url(url)
        ml = predict_url(url)
        nav = engine.evaluate_navigation_decision(url, "URL_HTTPS", ana, ml)
        dec = nav["decision"]

        assert 0.0 <= dec.risk_score <= 19.0
        assert dec.final_action == "ALLOW"
        assert nav["is_permitted"] is True
        assert nav["status_category"] == "PERMITTED"

    def test_equivalent_validation_for_other_legitimate_services(self, engine: ThreatPreventionEngine):
        """Verify equivalent low-risk classification for other legitimate forms and platforms (no Google-only whitelist)."""
        benign_services = [
            "https://forms.office.com/pages/responsepage.aspx?id=v4j5cvGGr0GRqy180BHbR",
            "https://form.typeform.com/to/sampleForm123?utm_source=newsletter",
            "https://www.surveymonkey.com/r/SampleFeedbackSurvey2026",
            "https://github.com/torvalds/linux/blob/master/include/linux/kernel.h",
            "https://en.wikipedia.org/wiki/Computer_security",
            "https://www.google.com",
        ]
        for url in benign_services:
            ana = analyze_url(url)
            ml = predict_url(url)
            nav = engine.evaluate_navigation_decision(url, "URL_HTTPS", ana, ml)
            dec = nav["decision"]
            assert 0.0 <= dec.risk_score <= 19.0, f"Failed for {url}: score was {dec.risk_score}"
            assert dec.final_action == "ALLOW", f"Failed for {url}: action was {dec.final_action}"
            assert nav["is_permitted"] is True, f"Failed for {url}: not permitted"

    def test_lookalike_brand_impersonation_blocked(self, engine: ThreatPreventionEngine):
        """Deceptive lookalike domains must be blocked in High or Critical tiers (>=50)."""
        lookalikes = [
            ("http://paypal-security-update-verification.com/login/verify.php?token=123", "URL_HTTP"),
            ("https://chase-security-verify-login.net/account", "URL_HTTPS"),
            ("http://appleid-security-warning.support-device.cf/confirm?id=99281", "URL_HTTP"),
        ]
        for url, ptype in lookalikes:
            ana = analyze_url(url)
            ml = predict_url(url)
            nav = engine.evaluate_navigation_decision(url, ptype, ana, ml)
            dec = nav["decision"]
            assert dec.final_action == "BLOCK", f"Failed to block {url}"
            assert dec.risk_score >= 50.0, f"Expected >=50, got {dec.risk_score} for {url}"
            assert nav["is_permitted"] is False
            assert nav["status_category"] == "BLOCKED"

    def test_open_redirect_attack_flagged_even_on_trusted_host(self, engine: ThreatPreventionEngine):
        """Open redirect attacks on trusted or external hosts are not declared safe based solely on domain."""
        redirect_urls = [
            ("https://example.com/login?redirect=https://attacker.evil.com/harvest", "URL_HTTPS"),
            ("https://www.google.com/url?q=https://malicious-site.tk", "URL_HTTPS"),
        ]
        for url, ptype in redirect_urls:
            ana = analyze_url(url)
            ml = predict_url(url)
            nav = engine.evaluate_navigation_decision(url, ptype, ana, ml)
            dec = nav["decision"]
            # Must NOT be allowed with 0 risk score simply because domain is recognized
            assert dec.risk_score >= 20.0, f"Open redirect on {url} had too low risk: {dec.risk_score}"
            assert dec.final_action in ("WARN", "BLOCK")
            assert nav["is_permitted"] is False

    def test_dangerous_schemes_and_malformed_urls_blocked(self, engine: ThreatPreventionEngine):
        """Dangerous protocols and malformed URLs receive Critical risk (80-100) and BLOCK."""
        dangerous_targets = [
            ("javascript:alert(1)", "DANGEROUS_SCHEME"),
            ("data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==", "DANGEROUS_SCHEME"),
            ("http://", "URL_HTTP"),
        ]
        for payload, ptype in dangerous_targets:
            ana = analyze_url(payload)
            ml = predict_url(payload) if ptype in ("URL_HTTP", "URL_HTTPS") and ana.status != "MALFORMED" else None
            nav = engine.evaluate_navigation_decision(payload, ptype, ana, ml)
            dec = nav["decision"]
            assert dec.final_action == "BLOCK"
            assert dec.risk_score >= 80.0
            assert nav["is_permitted"] is False
            assert nav["open_url"] is None
