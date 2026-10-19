"""
Comprehensive automated test suite for the QRShield Indicator-Based URL Threat Detection
and Centralized Deterministic Risk Scoring Engine.

Covers all 10 mandated requirements:
1. Legitimate Google Forms URLs
2. Other legitimate websites
3. Malformed URLs and deceptive hostnames
4. Homoglyph and typosquatting attacks
5. Known malicious URLs from safe test datasets
6. Redirect chains and suspicious destinations
7. Threat-intelligence timeouts and unavailable providers
8. Conflicting ML and rule-based predictions
9. Duplicate evidence and deterministic scores
10. Database write failures honest handling
"""

import pytest
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

from modules.url_analyzer import (
    analyze_url,
    detect_homoglyphs,
    detect_typosquatting,
    inspect_url_structure,
    inspect_safe_redirects,
    inspect_tls_certificate,
    URLAnalysisResult,
    IndicatorRecord,
)
from modules.threat_intelligence import (
    lookup_threat_intelligence,
    ThreatIntelAggregateResult,
    ThreatIntelProviderResult,
    STATUS_MATCH_MALICIOUS,
    STATUS_CLEAN,
    STATUS_UNAVAILABLE,
    STATUS_UNKNOWN,
    STATUS_RATE_LIMITED,
    STATUS_TIMEOUT,
)
from modules.threat_prevention import (
    evaluate_security_decision,
    ThreatPreventionEngine,
    SecurityDecision,
    get_prevention_engine,
)
from modules.ml_classifier import MLPredictionResult
from modules.database import save_complete_scan, get_last_error


# ==============================================================================
# 1. Legitimate Google Forms URLs
# ==============================================================================
class TestLegitimateGoogleForms:
    @pytest.fixture
    def engine(self):
        return get_prevention_engine()

    def test_standard_google_forms_viewform_low_risk(self, engine):
        """Standard docs.google.com/forms/d/e/.../viewform must be Low Risk (0-19) and ALLOW."""
        url = "https://docs.google.com/forms/d/e/1FAIpQLScP_7FjM5Y9z8nK1l4wX8yA0vE9qT1/viewform"
        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTPS",
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        assert 0.0 <= decision.risk_score <= 19.0, f"Expected Low Risk (0-19), got {decision.risk_score}"
        assert decision.classification == "SAFE"
        assert decision.final_enforcement_action == "ALLOW"
        assert decision.recommended_action in ("ALLOW", "WARN")

    def test_google_forms_with_query_params_not_penalized(self, engine):
        """Forms with tracking/prefill query parameters must not receive arbitrary penalties."""
        url = "https://docs.google.com/forms/d/e/1FAIpQLScP_7FjM5Y9z8nK1l4wX8yA0vE9qT1/viewform?usp=sf_link&entry.1234=test"
        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTPS",
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        assert decision.risk_score <= 19.0
        assert decision.final_enforcement_action == "ALLOW"

    def test_forms_gle_shortlink_low_risk(self, engine):
        """forms.gle shortlinks must evaluate cleanly in low-risk tier when structure is valid."""
        url = "https://forms.gle/7w5yBwJ5qC3h8k9A6"
        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTPS",
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        assert decision.risk_score <= 19.0
        assert decision.final_enforcement_action == "ALLOW"


# ==============================================================================
# 2. Other Legitimate Websites
# ==============================================================================
class TestLegitimateWebsites:
    @pytest.mark.parametrize("url", [
        "https://www.google.com",
        "https://en.wikipedia.org/wiki/QR_code",
        "https://github.com/torvalds/linux",
        "https://www.microsoft.com/en-us/windows",
        "https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final",
        "https://arxiv.org/abs/2301.00001",
        "https://forms.office.com/pages/responsepage.aspx?id=v4j5cvGGr0GRqy180BHbR",
        "https://form.typeform.com/to/sampleForm123",
    ])
    def test_benign_sites_evaluate_to_low_risk(self, url):
        """Legitimate websites must evaluate to Low Risk (0-19) with ALLOW navigation."""
        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTPS",
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        assert 0.0 <= decision.risk_score <= 19.0, f"Failed for {url}: score was {decision.risk_score}"
        assert decision.classification == "SAFE"
        assert decision.final_enforcement_action == "ALLOW"


# ==============================================================================
# 3. Malformed URLs and Deceptive Hostnames
# ==============================================================================
class TestMalformedUrlsAndDeceptiveHostnames:
    def test_deceptive_authority_embedded_credentials(self):
        """Embedded credentials in authority (user:pass@host) must trigger HIGH severity indicator."""
        url = "http://target-security-alert@192.168.0.1/login.php"
        records = inspect_url_structure(url, urlparse(url), urlparse(url).hostname or "")
        ind_ids = [r.indicator_id for r in records]
        assert "IND_DECEPTIVE_USERINFO" in ind_ids or "IND_IP_HOST" in ind_ids or "IND_PRIVATE_IP_HOST" in ind_ids

        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTP",
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        assert decision.risk_score >= 50.0
        assert decision.final_enforcement_action == "BLOCK"

    def test_backslash_host_confusion_flagged(self):
        """Backslashes in URL authority/path must trigger IND_MALFORMED_SYNTAX."""
        url = "http://legitimate-bank.com\\attacker.com/portal"
        records = inspect_url_structure(url, urlparse(url), urlparse(url).hostname or "")
        ind_ids = [r.indicator_id for r in records]
        assert "IND_MALFORMED_SYNTAX" in ind_ids

    def test_numerical_ip_host_flagged(self):
        """Raw IP address host must trigger IND_IP_HOST or IND_PRIVATE_IP_HOST."""
        url = "http://198.51.100.24/admin/update"
        records = inspect_url_structure(url, urlparse(url), urlparse(url).hostname or "")
        ind_ids = [r.indicator_id for r in records]
        assert "IND_IP_HOST" in ind_ids or "IND_PRIVATE_IP_HOST" in ind_ids

    def test_unusual_trojan_port_flagged(self):
        """Unusual ports (e.g. 4444, 6667, 23) must trigger IND_UNUSUAL_PORT."""
        url = "http://example.com:4444/shell"
        records = inspect_url_structure(url, urlparse(url), urlparse(url).hostname or "")
        ind_ids = [r.indicator_id for r in records]
        assert "IND_UNUSUAL_PORT" in ind_ids

    def test_dangerous_schemes_blocked(self):
        """Dangerous URI schemes (javascript:, data:, file:) must trigger CRITICAL and BLOCK."""
        for dangerous in ["javascript:alert(document.cookie)", "data:text/html,<script>steal()</script>", "file:///etc/passwd"]:
            decision = evaluate_security_decision(
                decoded_payload=dangerous,
                payload_type="DANGEROUS_SCHEME",
                enable_threat_intel=False,
                enable_live_redirects=False,
                enable_live_tls=False,
            )
            assert decision.final_enforcement_action == "BLOCK"
            assert decision.risk_score >= 80.0
            assert decision.classification == "MALICIOUS"


# ==============================================================================
# 4. Homoglyph and Typosquatting Attacks
# ==============================================================================
class TestHomoglyphAndTyposquatting:
    def test_cyrillic_homoglyph_detection(self):
        """Domain containing Cyrillic 'о' instead of Latin 'o' must be detected as homoglyph."""
        # gооgle.com with Cyrillic small letter o (\u043e)
        domain = "g\u043e\u043egle.com"
        records = detect_homoglyphs(domain)
        assert len(records) > 0
        assert records[0].indicator_id == "IND_HOMOGLYPH_DOMAIN"
        assert records[0].severity in ("CRITICAL", "HIGH")
        assert records[0].weight >= 40.0

    def test_punycode_homoglyph_detection(self):
        """IDN Punycode domain must be flagged as homoglyph attack."""
        domain = "xn--ggle-55da.com"
        records = detect_homoglyphs(domain)
        assert len(records) > 0
        assert records[0].indicator_id == "IND_HOMOGLYPH_DOMAIN"

    def test_typosquatting_levenshtein_distance(self):
        """Brand typo within Levenshtein distance 1-2 must trigger IND_TYPOSQUATTING."""
        typos = ["g00gle.com", "paypa1.com", "micros0ft.com", "app1e.com"]
        for domain in typos:
            records = detect_typosquatting(domain)
            assert len(records) > 0, f"Failed to detect typosquatting for {domain}"
            assert records[0].indicator_id in ("IND_TYPOSQUATTING", "IND_BRAND_IMPERSONATION")

    def test_brand_impersonation_on_untrusted_domain(self):
        """Brand name in subdomain or hyphenated domain must trigger IND_BRAND_IMPERSONATION."""
        domain = "paypal-account-center.update-billing.tk"
        records = detect_typosquatting(domain)
        assert len(records) > 0
        assert records[0].indicator_id == "IND_BRAND_IMPERSONATION"
        assert records[0].weight >= 40.0


# ==============================================================================
# 5. Known Malicious URLs From Safe Test Datasets
# ==============================================================================
class TestKnownMaliciousUrls:
    @pytest.fixture
    def engine(self):
        return get_prevention_engine()

    def test_known_phishing_url_blocked(self, engine):
        """Known phishing pattern must produce High/Critical risk score and BLOCK."""
        url = "http://login.paypal.com.customer-verification-update.xyz/webapps/mpp/home"
        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTP",
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        assert decision.risk_score >= 50.0
        assert decision.classification == "MALICIOUS"
        assert decision.final_enforcement_action == "BLOCK"

    def test_confirmed_threat_intel_match_mandates_block(self):
        """Confirmed threat intelligence match mandates BLOCK and >=90 risk score."""
        mock_ti = ThreatIntelAggregateResult(
            overall_status=STATUS_MATCH_MALICIOUS,
            is_malicious=True,
            primary_threat="URLhaus Malware Repository",
            evidence=["Malware download hosted on domain"],
            provider_results={
                "URLhaus": ThreatIntelProviderResult(
                    provider="URLhaus",
                    status=STATUS_MATCH_MALICIOUS,
                    is_malicious=True,
                    threat_type="malware_download",
                )
            },
        )
        with patch("modules.threat_prevention.lookup_threat_intelligence", return_value=mock_ti):
            decision = evaluate_security_decision(
                decoded_payload="http://malware-download-server.top/payload.exe",
                payload_type="URL_HTTP",
                enable_threat_intel=True,
                enable_live_redirects=False,
                enable_live_tls=False,
            )
            assert decision.risk_score >= 90.0
            assert decision.classification == "MALICIOUS"
            assert decision.final_enforcement_action == "BLOCK"
            assert decision.threat_intelligence_status == STATUS_MATCH_MALICIOUS


# ==============================================================================
# 6. Redirect Chains and Suspicious Destinations
# ==============================================================================
class TestRedirectChainsAndSuspiciousDestinations:
    def test_excessive_redirect_chain_flagged(self):
        """Chains with >= 4 redirects must trigger IND_EXCESSIVE_REDIRECTS."""
        mock_client = MagicMock()
        mock_resp_redir = MagicMock()
        mock_resp_redir.status_code = 302
        mock_resp_redir.headers = {"location": "https://hop-domain.com/next"}
        mock_client.head.return_value = mock_resp_redir

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value = mock_client
            records, details = inspect_safe_redirects("https://origin-url.com/start")
            ind_ids = [r.indicator_id for r in records]
            assert "IND_EXCESSIVE_REDIRECTS" in ind_ids

    def test_protocol_downgrade_redirect_flagged(self):
        """Redirect from HTTPS to cleartext HTTP must trigger IND_REDIRECT_PROTOCOL_DOWNGRADE."""
        mock_client = MagicMock()
        r1 = MagicMock()
        r1.status_code = 302
        r1.headers = {"location": "http://insecure-destination.com/plain"}
        r2 = MagicMock()
        r2.status_code = 200
        r2.headers = {}
        mock_client.head.side_effect = [r1, r2]

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value = mock_client
            records, details = inspect_safe_redirects("https://secure-login.com")
            ind_ids = [r.indicator_id for r in records]
            assert "IND_PROTOCOL_DOWNGRADE" in ind_ids

    def test_ssrf_private_ip_redirect_blocked(self):
        """Redirect attempting redirection to private RFC1918 / loopback IP must trigger IND_SSRF_REDIRECT."""
        mock_client = MagicMock()
        r1 = MagicMock()
        r1.status_code = 302
        r1.headers = {"location": "http://127.0.0.1/admin"}
        mock_client.head.return_value = r1

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value = mock_client
            records, details = inspect_safe_redirects("https://public-web-service.com/start")
            ind_ids = [r.indicator_id for r in records]
            assert "IND_SSRF_REDIRECT" in ind_ids


# ==============================================================================
# 7. Threat Intelligence Timeouts and Unavailable Providers
# ==============================================================================
class TestThreatIntelligenceTimeoutsAndUnavailableProviders:
    def test_timeout_handled_gracefully_not_clean(self):
        """API timeout must set STATUS_UNAVAILABLE or STATUS_UNKNOWN and NEVER STATUS_CLEAN."""
        import httpx
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectTimeout("Connection timeout")
            mock_client_cls.return_value.__enter__.return_value = mock_client

            res = lookup_threat_intelligence("https://example.com/check", enable_network=True)
            assert res.overall_status in (STATUS_UNAVAILABLE, STATUS_UNKNOWN, STATUS_TIMEOUT)
            assert res.overall_status != STATUS_CLEAN
            assert res.is_malicious is False

    def test_unavailable_threat_intel_does_not_certify_safety(self):
        """Decision engine must treat unavailable threat intel as unknown, not proof of safety."""
        mock_ti = ThreatIntelAggregateResult(
            overall_status=STATUS_UNAVAILABLE,
            is_malicious=False,
            primary_threat=None,
            evidence=[],
            provider_results={},
        )
        with patch("modules.threat_prevention.lookup_threat_intelligence", return_value=mock_ti):
            decision = evaluate_security_decision(
                decoded_payload="https://unknown-new-domain.xyz",
                payload_type="URL_HTTPS",
                enable_threat_intel=True,
                enable_live_redirects=False,
                enable_live_tls=False,
            )
            # Must record status as UNAVAILABLE
            assert decision.threat_intelligence_status == STATUS_UNAVAILABLE
            # Unknown reputation alone must not grant 100% confidence
            assert decision.confidence < 100.0


# ==============================================================================
# 8. Conflicting ML and Rule-Based Predictions
# ==============================================================================
class TestConflictingMLAndRuleBasedPredictions:
    def test_rule_overrides_optimistic_ml_on_high_risk_indicators(self):
        """If ML predicts 'Benign' (score 10) but structural indicators show IP host & userinfo, BLOCK."""
        url = "http://admin:secret@192.168.1.1/login"
        analysis = analyze_url(url)
        optimistic_ml = MLPredictionResult(
            predicted_label="Benign",
            predicted_class=0,
            risk_score=10.0,
            confidence=85.0,
            probabilities={"Benign": 0.90, "Malicious": 0.10},
            is_model_available=True,
        )

        decision = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=optimistic_ml,
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )

        # Safety rule must strictly enforce BLOCK despite optimistic ML
        assert decision.final_enforcement_action == "BLOCK"
        assert decision.risk_score >= 50.0

    def test_rl_cannot_override_mandatory_security_rules(self):
        """If RL agent recommends ALLOW for a malicious payload, mandatory constraint overrides to BLOCK."""
        engine = get_prevention_engine()
        url = "http://phishing-bank-login.com/account"
        analysis = analyze_url(url)
        malicious_ml = MLPredictionResult(
            predicted_label="Malicious",
            predicted_class=1,
            risk_score=90.0,
            confidence=95.0,
            probabilities={"Benign": 0.05, "Malicious": 0.95},
            is_model_available=True,
        )

        # Force RL table to recommend ALLOW
        state_key = engine.rl_agent.get_state_key(
            rf_pred="Malicious",
            rf_risk_score=90.0,
            heuristic_category=analysis.risk_category,
            confidence=95.0,
        )
        engine.rl_agent.initialize_state(state_key)
        engine.rl_agent.q_table[state_key]["ALLOW"] = 1000.0

        decision = engine.evaluate_threat(
            raw_payload=url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=malicious_ml,
        )

        assert decision.recommended_action == "ALLOW"  # RL suggested ALLOW
        assert decision.final_enforcement_action == "BLOCK"  # Security policy strictly overrode!
        assert decision.is_safety_override is True


# ==============================================================================
# 9. Duplicate Evidence and Deterministic Scoring
# ==============================================================================
class TestDuplicateEvidenceAndDeterministicScoring:
    def test_deduplicated_indicators_prevent_score_inflation(self):
        """Evaluating the same indicator repeatedly must NOT increase the risk score."""
        url = "http://198.51.100.1/admin"
        ana1 = analyze_url(url)

        # Build duplicate indicator list
        ana_dup = URLAnalysisResult(
            raw_payload=url,
            is_valid_url=True,
            status="VALID_WEB_URL",
            validation_message="Valid web destination.",
            normalized_url=url,
            scheme="http",
            hostname="198.51.100.1",
            path="/admin",
            indicator_records=ana1.indicator_records * 3,  # Triplicate indicators!
            risk_score=0.35,
        )

        dec1 = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTP",
            analysis_result=ana1,
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )
        dec_dup = evaluate_security_decision(
            decoded_payload=url,
            payload_type="URL_HTTP",
            analysis_result=ana_dup,
            enable_threat_intel=False,
            enable_live_redirects=False,
            enable_live_tls=False,
        )

        # Scores must be completely identical because evidence_map keys by indicator_id
        assert dec1.risk_score == dec_dup.risk_score
        assert len(dec1.evidence_list) == len(dec_dup.evidence_list)

    def test_deterministic_scoring_reproducibility(self):
        """Identical URL input must produce bit-for-bit identical risk score and classification."""
        url = "https://docs.google.com/forms/d/e/1FAIpQLScP_7FjM5Y9z8nK1l4wX8yA0vE9qT1/viewform"
        dec_a = evaluate_security_decision(decoded_payload=url, payload_type="URL_HTTPS", enable_threat_intel=False, enable_live_redirects=False, enable_live_tls=False)
        dec_b = evaluate_security_decision(decoded_payload=url, payload_type="URL_HTTPS", enable_threat_intel=False, enable_live_redirects=False, enable_live_tls=False)

        assert dec_a.risk_score == dec_b.risk_score
        assert dec_a.classification == dec_b.classification
        assert dec_a.confidence == dec_b.confidence
        assert dec_a.final_enforcement_action == dec_b.final_enforcement_action


# ==============================================================================
# 10. Database Write Failures Honest Handling
# ==============================================================================
class TestDatabaseWriteFailuresHonestHandling:
    def test_database_write_failure_returns_none_and_records_error(self):
        """When Supabase insert throws an exception, save_complete_scan must return None honestly."""
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("Simulated Supabase Postgrest API Failure")

        from modules.threat_prevention import SecurityDecision
        decision = SecurityDecision(
            decoded_payload="https://example.com",
            normalized_url="https://example.com",
            risk_score=10.0,
            confidence=90.0,
            classification="SAFE",
            evidence_list=[],
            threat_intelligence_status="UNKNOWN",
            recommended_action="ALLOW",
            final_enforcement_action="ALLOW",
            payload_type="URL_HTTPS",
            explanation="Safe navigation permitted.",
            is_safety_override=False,
        )

        res = save_complete_scan(
            extracted_url="https://example.com",
            qr_payload_type="URL_HTTPS",
            predicted_class="SAFE",
            risk_score=0.10,
            decision_dict={"decision": decision},
            client=mock_client,
        )

        assert res is None
        err = get_last_error()
        assert err is not None
        assert "Simulated Supabase Postgrest API Failure" in str(err) or "Exception" in str(err)

