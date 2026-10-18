"""Unit tests for the Threat Prevention Module.

Tests:
- Deterministic safety overrides on malicious or high-threat signals.
- Unconditional blocking of malformed and dangerous URI schemes.
- Specialized scheme user confirmation warnings.
- Plain text safe handling.
- Baseline policy comparison execution.
"""

import pytest

from modules.ml_classifier import MLPredictionResult
from modules.threat_prevention import (
    PreventionDecision,
    SafetyConstraintsConfig,
    ThreatPreventionEngine,
    get_default_benchmark_scenarios,
    get_prevention_engine,
)
from modules.url_analyzer import URLAnalysisResult, analyze_url


class TestSafetyConstraints:
    """Test suite for safety rules and deterministic constraints."""

    @pytest.fixture
    def engine(self):
        """Initialized threat prevention engine."""
        return ThreatPreventionEngine()

    def test_block_dangerous_uri_scheme(self, engine: ThreatPreventionEngine):
        """Verify dangerous schemes (e.g. javascript:) are unconditionally BLOCKED."""
        payload = "javascript:alert('xss')"
        analysis = analyze_url(payload)

        decision = engine.evaluate_threat(
            raw_payload=payload,
            payload_type="DANGEROUS_SCHEME",
            analysis_result=analysis,
        )

        assert isinstance(decision, PreventionDecision)
        assert decision.final_action == "BLOCK"
        assert decision.is_safety_override is True
        assert decision.safety_rule_applied == "CRITICAL_DANGEROUS_OR_MALFORMED"

    def test_block_malformed_url(self, engine: ThreatPreventionEngine):
        """Verify malformed URLs with unparseable format are unconditionally BLOCKED."""
        payload = "https://[invalid-ipv6-bracket"
        analysis = analyze_url(payload)

        decision = engine.evaluate_threat(
            raw_payload=payload,
            payload_type="URL_HTTPS",
            analysis_result=analysis,
        )

        assert decision.final_action == "BLOCK"
        assert decision.is_safety_override is True

    def test_warn_specialized_scheme(self, engine: ThreatPreventionEngine):
        """Verify specialized schemes (tel:, mailto:) require user confirmation (WARN)."""
        payload = "tel:+1234567890"
        analysis = analyze_url(payload)

        decision = engine.evaluate_threat(
            raw_payload=payload,
            payload_type="SPECIALIZED_SCHEME",
            analysis_result=analysis,
        )

        assert decision.final_action == "WARN"
        assert decision.safety_rule_applied == "RULE_SPECIALIZED_PROTOCOL"

    def test_allow_plain_text(self, engine: ThreatPreventionEngine):
        """Verify plain text payloads without URLs are safely allowed to view."""
        payload = "Table 14 - Guest Wifi Password: securePass123"
        analysis = analyze_url(payload)

        decision = engine.evaluate_threat(
            raw_payload=payload,
            payload_type="PLAIN_TEXT",
            analysis_result=analysis,
        )

        assert decision.final_action == "ALLOW"
        assert decision.risk_score == 0.0

    def test_safety_override_when_rl_proposes_allow_on_malicious(self, engine: ThreatPreventionEngine):
        """Test safety constraint: If RL recommends ALLOW but ML is Malicious, force BLOCK."""
        payload = "http://phishing-bank-login.com/account"
        analysis = analyze_url(payload)

        # Mock ML prediction as Malicious with High Risk
        ml_pred = MLPredictionResult(
            predicted_label="Malicious",
            predicted_class=1,
            risk_score=85.0,
            confidence=92.0,
            probabilities={"Benign": 0.08, "Malicious": 0.92},
            is_model_available=True,
        )

        # Force RL table to recommend ALLOW for this state
        state_key = engine.rl_agent.get_state_key(
            rf_pred="Malicious",
            rf_risk_score=85.0,
            heuristic_category=analysis.risk_category,
            confidence=92.0,
        )
        engine.rl_agent.initialize_state(state_key)
        engine.rl_agent.q_table[state_key]["ALLOW"] = 999.0
        engine.rl_agent.q_table[state_key]["BLOCK"] = -999.0

        decision = engine.evaluate_threat(
            raw_payload=payload,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        # RL suggested ALLOW, but safety policy must override to BLOCK!
        assert decision.raw_action == "ALLOW"
        assert decision.final_action == "BLOCK"
        assert decision.is_safety_override is True
        assert "SAFETY_OVERRIDE" in (decision.safety_rule_applied or "")

    def test_clean_benign_url_allowed(self, engine: ThreatPreventionEngine):
        """Verify clearly benign URL with low risk is ALLOWED."""
        payload = "https://www.wikipedia.org"
        analysis = analyze_url(payload)

        ml_pred = MLPredictionResult(
            predicted_label="Benign",
            predicted_class=0,
            risk_score=5.0,
            confidence=98.0,
            probabilities={"Benign": 0.98, "Malicious": 0.02},
            is_model_available=True,
        )

        decision = engine.evaluate_threat(
            raw_payload=payload,
            payload_type="URL_HTTPS",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        assert decision.final_action == "ALLOW"
        assert decision.is_safety_override is False


class TestPolicyBenchmarking:
    """Test suite for comparing Q-Learning with fixed-threshold baseline."""

    def test_benchmark_policies_execution(self):
        """Verify benchmark comparison runs and reports all required metrics."""
        engine = get_prevention_engine()
        scenarios = get_default_benchmark_scenarios()

        bench = engine.benchmark_policies(scenarios)

        assert "fixed_threshold_policy" in bench
        assert "q_learning_adaptive_policy" in bench

        fixed = bench["fixed_threshold_policy"]
        rl = bench["q_learning_adaptive_policy"]

        for metric in ["malicious_allow_rate", "benign_block_rate", "warning_rate", "optimal_decision_rate"]:
            assert metric in fixed
            assert metric in rl
            assert isinstance(fixed[metric], (int, float))
            assert isinstance(rl[metric], (int, float))


class TestSafeNavigationDecision:
    """Test suite verifying safe destination opening rules and centralized navigation decision."""

    @pytest.fixture
    def engine(self):
        return ThreatPreventionEngine()

    def test_permitted_google_forms_url(self, engine: ThreatPreventionEngine):
        """Verify legitimate Google Forms URL is permitted and exposes 'Open Google Form'."""
        url = "https://forms.gle/sampleSurvey123"
        analysis = analyze_url(url)
        ml_pred = MLPredictionResult(
            predicted_label="Benign",
            predicted_class=0,
            risk_score=2.0,
            confidence=95.0,
            probabilities={"Benign": 0.95, "Malicious": 0.05},
            is_model_available=True,
        )

        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTPS",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        assert nav["is_permitted"] is True
        assert nav["status_category"] == "PERMITTED"
        assert nav["open_url"] == url
        assert nav["button_label"] == "🌐 Open Google Form"
        assert nav["is_google_form"] is True

    def test_benign_https_url(self, engine: ThreatPreventionEngine):
        """Verify normal benign HTTPS URL is permitted and exposes 'Open Website'."""
        url = "https://www.wikipedia.org"
        analysis = analyze_url(url)
        ml_pred = MLPredictionResult(
            predicted_label="Benign",
            predicted_class=0,
            risk_score=5.0,
            confidence=98.0,
            probabilities={"Benign": 0.98, "Malicious": 0.02},
            is_model_available=True,
        )

        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTPS",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        assert nav["is_permitted"] is True
        assert nav["status_category"] == "PERMITTED"
        assert nav["open_url"] == url
        assert nav["button_label"] == "🌐 Open Website"
        assert nav["is_google_form"] is False

    def test_malicious_url_blocked_no_open_button(self, engine: ThreatPreventionEngine):
        """Verify malicious URL is BLOCKED and never exposes an opening button."""
        url = "http://phishing-banking-portal.com/login"
        analysis = analyze_url(url)
        ml_pred = MLPredictionResult(
            predicted_label="Malicious",
            predicted_class=1,
            risk_score=90.0,
            confidence=94.0,
            probabilities={"Benign": 0.06, "Malicious": 0.94},
            is_model_available=True,
        )

        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        assert nav["is_permitted"] is False
        assert nav["status_category"] == "BLOCKED"
        assert nav["open_url"] is None
        assert nav["button_label"] is None
        assert "Blocked" in nav["display_title"]

    def test_uncertain_url_no_open_button(self, engine: ThreatPreventionEngine):
        """Verify uncertain / warning URL does not provide direct opening button."""
        url = "http://unknown-new-domain.info/verify"
        analysis = analyze_url(url)
        # Medium risk scenario
        analysis.risk_category = "MEDIUM"
        analysis.risk_score = 0.45

        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=None,
        )

        assert nav["is_permitted"] is False
        assert nav["status_category"] == "UNCERTAIN"
        assert nav["open_url"] is None
        assert nav["button_label"] is None

    def test_invalid_url_blocked(self, engine: ThreatPreventionEngine):
        """Verify invalid or malformed URL is blocked unconditionally."""
        url = "https://[invalid-ipv6-bracket"
        analysis = analyze_url(url)

        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTPS",
            analysis_result=analysis,
        )

        assert nav["is_permitted"] is False
        assert nav["status_category"] == "BLOCKED"
        assert nav["open_url"] is None
        assert nav["button_label"] is None

    def test_unsupported_uri_scheme_blocked(self, engine: ThreatPreventionEngine):
        """Verify unsupported dangerous schemes (javascript:, data:, file:) are blocked."""
        for scheme_payload in ["javascript:alert('pwned')", "data:text/html,<script>alert(1)</script>", "file:///etc/passwd"]:
            analysis = analyze_url(scheme_payload)
            nav = engine.evaluate_navigation_decision(
                raw_payload=scheme_payload,
                payload_type="DANGEROUS_SCHEME",
                analysis_result=analysis,
            )

            assert nav["is_permitted"] is False
            assert nav["status_category"] == "BLOCKED"
            assert nav["open_url"] is None
            assert nav["button_label"] is None

    def test_plain_text_payload_not_treated_as_url(self, engine: ThreatPreventionEngine):
        """Verify plain text payloads are safe to view but never openable as a URL."""
        text_payload = "Wi-Fi: SSID=OfficeGuest; P=SuperSecret123"
        analysis = analyze_url(text_payload)

        nav = engine.evaluate_navigation_decision(
            raw_payload=text_payload,
            payload_type="PLAIN_TEXT",
            analysis_result=analysis,
        )

        assert nav["is_permitted"] is False
        assert nav["status_category"] == "INFO"
        assert nav["open_url"] is None
        assert nav["button_label"] is None
        assert "Plain Text" in nav["display_title"]

    def test_high_risk_url_with_rl_allow_is_blocked(self, engine: ThreatPreventionEngine):
        """Verify safety enforcer overrides Q-learning when RL recommends ALLOW on high-risk input."""
        url = "http://malicious-credential-harvest.com/login"
        analysis = analyze_url(url)
        ml_pred = MLPredictionResult(
            predicted_label="Malicious",
            predicted_class=1,
            risk_score=95.0,
            confidence=96.0,
            probabilities={"Benign": 0.04, "Malicious": 0.96},
            is_model_available=True,
        )

        # Force Q-table to recommend ALLOW
        state_key = engine.rl_agent.get_state_key(
            rf_pred="Malicious",
            rf_risk_score=95.0,
            heuristic_category=analysis.risk_category,
            confidence=96.0,
        )
        engine.rl_agent.initialize_state(state_key)
        engine.rl_agent.q_table[state_key]["ALLOW"] = 1000.0
        engine.rl_agent.q_table[state_key]["BLOCK"] = -1000.0

        nav = engine.evaluate_navigation_decision(
            raw_payload=url,
            payload_type="URL_HTTP",
            analysis_result=analysis,
            ml_prediction=ml_pred,
        )

        # Must be blocked and no opening button exposed!
        assert nav["decision"].raw_action == "ALLOW"
        assert nav["decision"].final_action == "BLOCK"
        assert nav["decision"].is_safety_override is True
        assert nav["is_permitted"] is False
        assert nav["open_url"] is None
        assert nav["button_label"] is None
        assert nav["status_category"] == "BLOCKED"

