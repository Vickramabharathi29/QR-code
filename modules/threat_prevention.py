"""Threat Prevention Module for QRShield.

Translates reinforcement learning decisions into concrete threat prevention actions,
applies deterministic security constraints, and provides a unified, indicator-based
threat detection and decision engine.

Safety Constraints:
- Malformed or unsupported URLs are unconditionally BLOCKED.
- High-risk URLs (risk >= 50 or confirmed threat intel) are NEVER allowed, even if RL recommends ALLOW.
- Model uncertainty or unverified transport triggers conservative fallback (WARN or BLOCK).
- Confirmed malicious destinations can NEVER be opened automatically.
"""

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import config
from modules.qr_decoder import classify_payload
from modules.reinforcement_learning import (
    ACTIONS,
    QLearningAgent,
    RewardConfig,
    compute_reward,
    encode_state,
    get_rl_agent,
)
from modules.threat_intelligence import (
    STATUS_CLEAN,
    STATUS_MATCH_MALICIOUS,
    STATUS_RATE_LIMITED,
    STATUS_TIMEOUT,
    STATUS_UNAVAILABLE,
    STATUS_UNKNOWN,
    lookup_threat_intelligence,
)
from modules.url_analyzer import (
    URLAnalysisResult,
    analyze_url,
    inspect_safe_redirects,
    inspect_tls_certificate,
    is_private_or_local_host,
)

logger = logging.getLogger("qrshield.prevention")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class SafetyConstraintsConfig:
    """Configurable safety rules applied on top of reinforcement learning recommendations."""

    block_malformed_urls: bool = True
    block_dangerous_schemes: bool = True
    prevent_allow_on_high_risk: bool = True
    prevent_allow_on_malicious_ml: bool = True
    prevent_allow_on_uncertain_medium: bool = True
    prevent_allow_on_unavailable_model: bool = True
    high_risk_threshold: float = 50.0
    medium_risk_threshold: float = 20.0


@dataclass
class SecurityDecision:
    """Standardized outcome produced by the centralized threat detection and decision engine.

    Satisfies the requirement for a single unified decision model shared across
    the scanner, ML model, RL agent, UI, and Supabase persistence layer.
    """

    decoded_payload: str
    normalized_url: Optional[str]
    risk_score: float                   # 0.0 to 100.0 (deterministic)
    confidence: float                   # 0.0 to 100.0
    classification: str                 # 'SAFE', 'SUSPICIOUS', 'MALICIOUS', 'UNVERIFIED'
    evidence_list: List[Dict[str, Any]] # Structured evidence items
    threat_intelligence_status: str     # MATCH_MALICIOUS, CLEAN, UNAVAILABLE, TIMEOUT, etc.
    recommended_action: str             # Q-learning suggested action ("ALLOW", "WARN", "BLOCK")
    final_enforcement_action: str       # Security-enforced action ("ALLOW", "WARN", "BLOCK")
    payload_type: str = "URL_HTTPS"
    explanation: str = ""
    is_safety_override: bool = False
    override_reason: Optional[str] = None
    state_key: Optional[str] = None
    q_values: Dict[str, float] = field(default_factory=dict)
    safety_rule_applied: Optional[str] = None
    threat_intel_details: Dict[str, Any] = field(default_factory=dict)
    redirect_details: Optional[Dict[str, Any]] = None
    tls_details: Optional[Dict[str, Any]] = None

    @property
    def is_permitted(self) -> bool:
        """True if and only if final policy decision is ALLOW."""
        return self.final_enforcement_action == "ALLOW"

    @property
    def is_blocked(self) -> bool:
        """True if final policy decision is BLOCK."""
        return self.final_enforcement_action == "BLOCK"

    @property
    def is_uncertain(self) -> bool:
        """True if final policy decision is WARN."""
        return self.final_enforcement_action == "WARN"

    # Backward-compatibility aliases with legacy PreventionDecision
    @property
    def raw_action(self) -> str:
        return self.recommended_action

    @property
    def final_action(self) -> str:
        return self.final_enforcement_action

    @property
    def predicted_class(self) -> str:
        return self.classification

    @property
    def heuristic_category(self) -> str:
        return self.classification

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to serializable dictionary."""
        d = asdict(self)
        d["is_permitted"] = self.is_permitted
        d["is_blocked"] = self.is_blocked
        d["is_uncertain"] = self.is_uncertain
        d["raw_action"] = self.raw_action
        d["final_action"] = self.final_action
        d["predicted_class"] = self.predicted_class
        d["heuristic_category"] = self.heuristic_category
        return d


# Alias for backward compatibility
PreventionDecision = SecurityDecision


def evaluate_security_decision(
    decoded_payload: str,
    payload_type: Optional[str] = None,
    analysis_result: Optional[URLAnalysisResult] = None,
    ml_prediction: Optional[Any] = None,
    enable_threat_intel: bool = True,
    enable_live_redirects: bool = True,
    enable_live_tls: bool = True,
    rl_agent: Optional[QLearningAgent] = None,
    safety_config: Optional[SafetyConstraintsConfig] = None,
) -> SecurityDecision:
    """Centralized, deterministic indicator-based threat detection and decision function.

    Used uniformly by the scanner, ML model, RL agent, UI, and Supabase persistence.

    Args:
        decoded_payload: Raw QR payload or target URL.
        payload_type: Optional payload classification from QR decoder.
        analysis_result: Optional precomputed static URL analysis result.
        ml_prediction: Optional precomputed ML prediction result.
        enable_threat_intel: Look up threat intelligence if applicable.
        enable_live_redirects: Perform bounded live redirect inspection.
        enable_live_tls: Perform TLS probe if HTTPS.
        rl_agent: Optional custom Q-learning agent.
        safety_config: Optional custom safety constraints configuration.

    Returns:
        SecurityDecision: Standardized decision object containing all required fields.
    """
    clean_payload = (decoded_payload or "").strip()
    agent = rl_agent or get_rl_agent()
    cfg = safety_config or SafetyConstraintsConfig()

    # 1. Classify QR payload type if not provided
    if not payload_type:
        ptype, _, _ = classify_payload(clean_payload)
    else:
        ptype = payload_type

    # --- CASE A: Non-Web Payloads (Privacy Safeguard: Never query remote threat intel) ---
    if ptype == "CONTACT_INFO":
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=None,
            risk_score=0.0,
            confidence=100.0,
            classification="SAFE",
            evidence_list=[{
                "indicator_id": "PAYLOAD_CONTACT_INFO",
                "severity": "INFO",
                "weight": 0.0,
                "category": "QR_PAYLOAD",
                "description": "Offline contact card payload (vCard / MeCard). No network navigation triggered.",
            }],
            threat_intelligence_status="SKIPPED_OFFLINE_PAYLOAD",
            recommended_action="ALLOW",
            final_enforcement_action="ALLOW",
            payload_type=ptype,
            explanation="Safe offline contact card. Payload contains address book information without remote navigation.",
            q_values={"ALLOW": 20.0, "WARN": -5.0, "BLOCK": -50.0},
        )

    if ptype == "WIFI_CONFIG":
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=None,
            risk_score=15.0,
            confidence=95.0,
            classification="SUSPICIOUS",
            evidence_list=[{
                "indicator_id": "PAYLOAD_WIFI_CONFIG",
                "severity": "MEDIUM",
                "weight": 15.0,
                "category": "QR_PAYLOAD",
                "description": "Wi-Fi network configuration QR code. Exercise caution when connecting to untrusted networks.",
            }],
            threat_intelligence_status="SKIPPED_OFFLINE_PAYLOAD",
            recommended_action="WARN",
            final_enforcement_action="WARN",
            payload_type=ptype,
            explanation="Caution advisory: Wi-Fi network credential payload. Verify network SSID before connecting your device.",
            q_values={"ALLOW": -10.0, "WARN": 15.0, "BLOCK": -20.0},
        )

    if ptype == "PAYMENT_PAYLOAD":
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=None,
            risk_score=25.0,
            confidence=95.0,
            classification="SUSPICIOUS",
            evidence_list=[{
                "indicator_id": "PAYLOAD_PAYMENT",
                "severity": "MEDIUM",
                "weight": 25.0,
                "category": "QR_PAYLOAD",
                "description": "Digital payment or cryptocurrency transaction transfer request. Verify recipient address and amount.",
            }],
            threat_intelligence_status="SKIPPED_OFFLINE_PAYLOAD",
            recommended_action="WARN",
            final_enforcement_action="WARN",
            payload_type=ptype,
            explanation="Payment verification required: Digital asset/payment transfer request. Explicit confirmation required before transfer.",
            q_values={"ALLOW": -15.0, "WARN": 20.0, "BLOCK": -15.0},
        )

    if ptype in ("SPECIALIZED_SCHEME", "UNSUPPORTED_SCHEME"):
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=None,
            risk_score=25.0,
            confidence=95.0,
            classification="SUSPICIOUS",
            evidence_list=[{
                "indicator_id": "RULE_SPECIALIZED_PROTOCOL",
                "severity": "MEDIUM",
                "weight": 25.0,
                "category": "QR_PAYLOAD",
                "description": "Specialized device protocol (tel:, mailto:, sms:, wifi:, etc.) detected. User confirmation required before launching external handler.",
            }],
            threat_intelligence_status="SKIPPED_OFFLINE_PAYLOAD",
            recommended_action="WARN",
            final_enforcement_action="WARN",
            payload_type=ptype,
            explanation="Caution required: Specialized device action detected. Requires user acknowledgement before launching native application handlers.",
            is_safety_override=True,
            override_reason="Safety Constraint: Specialized URI protocols (tel:, mailto:, wifi:) require explicit user confirmation.",
            q_values={"ALLOW": -10.0, "WARN": 10.0, "BLOCK": -20.0},
            safety_rule_applied="RULE_SPECIALIZED_PROTOCOL",
        )

    if ptype in ("PLAIN_TEXT", "EMPTY"):
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=None,
            risk_score=0.0,
            confidence=100.0,
            classification="SAFE",
            evidence_list=[{
                "indicator_id": "PAYLOAD_PLAIN_TEXT",
                "severity": "INFO",
                "weight": 0.0,
                "category": "QR_PAYLOAD",
                "description": "Plain text QR payload without active URI scheme or navigation capability.",
            }],
            threat_intelligence_status="SKIPPED_OFFLINE_PAYLOAD",
            recommended_action="ALLOW",
            final_enforcement_action="ALLOW",
            payload_type=ptype,
            explanation="Safe to view. The QR payload contains static text with no active network destination.",
            q_values={"ALLOW": 15.0, "WARN": -5.0, "BLOCK": -50.0},
        )

    if ptype == "DANGEROUS_SCHEME":
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=None,
            risk_score=100.0,
            confidence=100.0,
            classification="MALICIOUS",
            evidence_list=[{
                "indicator_id": "IND_DANGEROUS_SCHEME",
                "severity": "CRITICAL",
                "weight": 100.0,
                "category": "QR_PAYLOAD",
                "description": "Weaponized URI scheme detected (e.g. javascript:, data:, file:). Navigation strictly blocked.",
            }],
            threat_intelligence_status="SKIPPED_DANGEROUS_PAYLOAD",
            recommended_action="BLOCK",
            final_enforcement_action="BLOCK",
            payload_type=ptype,
            explanation="Critical Threat Neutralized: Dangerous URI scheme detected. Navigation strictly blocked to prevent client exploitation.",
            is_safety_override=True,
            override_reason="Critical Safety Constraint: Payload contains a dangerous URI scheme or malformed structure.",
            q_values={"ALLOW": -100.0, "WARN": -50.0, "BLOCK": 30.0},
            safety_rule_applied="CRITICAL_DANGEROUS_OR_MALFORMED",
        )

    # --- CASE B: Web Destination Analysis (URL_HTTPS / URL_HTTP) ---
    analysis = analysis_result or analyze_url(clean_payload)
    norm_url = analysis.normalized_url or clean_payload

    # Handle malformed URL structure
    if "MALFORMED" in analysis.status or analysis.status in ("MISSING_HOSTNAME", "INVALID_PORT") or not analysis.is_valid_url:
        return SecurityDecision(
            decoded_payload=clean_payload,
            normalized_url=norm_url,
            risk_score=85.0,
            confidence=95.0,
            classification="MALICIOUS",
            evidence_list=[{
                "indicator_id": "IND_MALFORMED_SYNTAX",
                "severity": "HIGH",
                "weight": 85.0,
                "category": "URL_STRUCTURE",
                "description": analysis.validation_message or "Malformed URL syntax or deceptive hostname structure.",
            }],
            threat_intelligence_status="SKIPPED_MALFORMED",
            recommended_action="BLOCK",
            final_enforcement_action="BLOCK",
            payload_type=ptype,
            explanation=f"Malformed URL Rejected: {analysis.validation_message}. Deceptive syntax is blocked unconditionally.",
            is_safety_override=True,
            override_reason="Mandatory Security Constraint: Malformed URL syntax.",
            q_values={"ALLOW": -80.0, "WARN": -40.0, "BLOCK": 25.0},
            safety_rule_applied="CRITICAL_MALFORMED_URL",
        )

    # 2. Collect Deduplicated Indicators into Evidence Map
    evidence_map: Dict[str, Dict[str, Any]] = {}
    for ir in analysis.indicator_records:
        evidence_map[ir.indicator_id] = ir.to_dict()

    # 3. Optional Bounded Live Redirect Inspection
    redirect_details = None
    if enable_live_redirects and analysis.is_valid_url and norm_url:
        try:
            redir_recs, r_details = inspect_safe_redirects(norm_url)
            redirect_details = r_details
            for rr in redir_recs:
                evidence_map[rr.indicator_id] = rr.to_dict()
        except Exception as exc:
            logger.debug("Safe redirect inspection error: %s", exc)

    # 4. Optional Safe TLS Certificate Inspection
    tls_details = None
    if enable_live_tls and analysis.scheme == "https" and analysis.hostname:
        try:
            tls_recs, t_details = inspect_tls_certificate(analysis.hostname)
            tls_details = t_details
            for tr in tls_recs:
                evidence_map[tr.indicator_id] = tr.to_dict()
        except Exception as exc:
            logger.debug("TLS inspection error: %s", exc)

    # 5. Modular Threat Intelligence Query
    ti_status = STATUS_UNKNOWN
    ti_details = {}
    is_ti_malicious = False

    if enable_threat_intel and analysis.is_valid_url and norm_url:
        try:
            ti_res = lookup_threat_intelligence(norm_url, enable_network=True)
            ti_status = ti_res.overall_status
            ti_details = ti_res.to_dict()
            is_ti_malicious = ti_res.is_malicious

            if is_ti_malicious or ti_res.overall_status == STATUS_MATCH_MALICIOUS:
                evidence_map["IND_THREAT_INTEL_MALICIOUS"] = {
                    "indicator_id": "IND_THREAT_INTEL_MALICIOUS",
                    "category": "REPUTATION",
                    "severity": "CRITICAL",
                    "weight": 90.0,
                    "description": f"Confirmed malicious threat intelligence match: {ti_res.primary_threat or 'Known phishing/malware repository'}.",
                    "evidence_data": ti_details,
                }
            elif ti_res.overall_status == STATUS_CLEAN:
                evidence_map["IND_THREAT_INTEL_CLEAN"] = {
                    "indicator_id": "IND_THREAT_INTEL_CLEAN",
                    "category": "REPUTATION",
                    "severity": "INFO",
                    "weight": 0.0,
                    "description": "Clean threat intelligence verification across reporting providers (URLhaus, Safe Browsing, VirusTotal).",
                    "evidence_data": ti_details,
                }
            else:
                evidence_map["IND_THREAT_INTEL_UNKNOWN"] = {
                    "indicator_id": "IND_THREAT_INTEL_UNKNOWN",
                    "category": "REPUTATION",
                    "severity": "INFO",
                    "weight": 0.0,
                    "description": f"Threat intelligence status: {ti_res.overall_status.lower()} (unavailability is not treated as proof of safety).",
                    "evidence_data": ti_details,
                }
        except Exception as exc:
            logger.debug("Threat intelligence query error: %s", exc)
            ti_status = STATUS_UNAVAILABLE

    # 6. Machine Learning Feature Inference
    ml_pred_result = ml_prediction
    if ml_pred_result is None and analysis_result is None and analysis.is_valid_url and norm_url:
        try:
            from modules.ml_classifier import predict_url
            ml_pred_result = predict_url(norm_url)
        except Exception as exc:
            logger.debug("ML inference error: %s", exc)

    is_model_available = False
    rf_pred = "Unknown"
    rf_risk = 0.0
    rf_conf = 50.0

    if ml_pred_result is not None and getattr(ml_pred_result, "is_model_available", False):
        is_model_available = True
        rf_pred = ml_pred_result.predicted_label
        rf_risk = float(ml_pred_result.risk_score)
        rf_conf = float(ml_pred_result.confidence)

        if rf_pred == "Malicious" and rf_risk >= 70.0:
            evidence_map["IND_ML_MALICIOUS"] = {
                "indicator_id": "IND_ML_MALICIOUS",
                "category": "PHISHING",
                "severity": "HIGH",
                "weight": 35.0,
                "description": f"Machine Learning classifier identified malicious lexical patterns (Confidence: {rf_conf:.1f}%).",
                "evidence_data": {"confidence": rf_conf, "risk_score": rf_risk},
            }

    # 7. Calculate Central Deterministic Risk Score
    # Sum deduplicated indicator weights
    raw_score = sum(float(ind.get("weight", 0.0)) for ind in evidence_map.values())
    if analysis_result is not None and getattr(analysis_result, "risk_score", 0.0):
        raw_score = max(raw_score, float(analysis_result.risk_score) * 100.0)
    if is_ti_malicious:
        raw_score = max(raw_score, 90.0)

    final_risk_score = round(max(0.0, min(100.0, raw_score)), 1)

    # 8. Determine Classification
    # Canonical thresholds:
    # 0–19: Low risk (SAFE)
    # 20–49: Suspicious (SUSPICIOUS or UNVERIFIED)
    # 50–79: High risk (MALICIOUS)
    # 80–100: Critical risk (MALICIOUS)
    if final_risk_score >= 50.0:
        classification = "MALICIOUS"
    elif final_risk_score >= 20.0:
        classification = "SUSPICIOUS"
    elif analysis.scheme == "http":
        classification = "UNVERIFIED"
    else:
        classification = "SAFE"

    # 9. Evidence Coverage & Confidence Metric
    coverage_score = 50.0
    if is_model_available:
        coverage_score += 20.0
    if enable_threat_intel and ti_status in (STATUS_CLEAN, STATUS_MATCH_MALICIOUS):
        coverage_score += 20.0
    if analysis.scheme == "https":
        coverage_score += 10.0
    final_confidence = round(min(100.0, max(50.0, coverage_score)), 1)

    # 10. Reinforcement Learning Policy Evaluation
    rl_state_risk = rf_risk if (ml_prediction and is_model_available) else final_risk_score
    rl_state_cat = analysis.risk_category if hasattr(analysis, "risk_category") else classification
    rl_state_conf = rf_conf if (ml_prediction and is_model_available) else final_confidence

    state_key = agent.get_state_key(
        rf_pred=rf_pred,
        rf_risk_score=rl_state_risk,
        heuristic_category=rl_state_cat,
        confidence=rl_state_conf,
        is_model_available=is_model_available,
    )
    raw_action = agent.choose_action(state_key, training=False)
    q_values = agent.get_q_values(state_key)


    # 11. Enforce Mandatory Security Constraints (RL cannot override safety rules)
    final_action = raw_action
    is_override = False
    override_reason = None
    safety_rule = None

    # Mandatory Rule 1: Confirmed malicious threat intel or High/Critical risk MUST be BLOCKED
    if (final_risk_score >= cfg.high_risk_threshold or classification == "MALICIOUS" or is_ti_malicious) and final_action != "BLOCK":
        final_action = "BLOCK"
        is_override = True
        override_reason = f"Mandatory Security Constraint: Confirmed threat or elevated risk score ({final_risk_score:.1f}/100) mandates BLOCK."
        safety_rule = "SAFETY_OVERRIDE_HIGH_THREAT"

    # Conservative Rule 2: Suspicious or unverified destination cannot be ALLOW
    elif final_action == "ALLOW" and (final_risk_score >= cfg.medium_risk_threshold or classification in ("SUSPICIOUS", "UNVERIFIED") or not is_model_available):
        final_action = "WARN"
        is_override = True
        override_reason = f"Safety Prudence: Elevated risk ({final_risk_score:.1f}/100) or unverified transport requires explicit confirmation (WARN)."
        safety_rule = "SAFETY_PRUDENCE_UNCERTAIN"

    # Permitted Rule 3: Clean, verified low risk
    elif final_risk_score < cfg.medium_risk_threshold and classification == "SAFE":
        final_action = "ALLOW"

    # Formulate Decision Explanation
    int_score = int(round(final_risk_score))
    if final_action == "BLOCK":
        severity_label = "Critical Threat" if final_risk_score >= 80.0 else "High Threat"
        explanation = f"Threat Neutralized: {severity_label} detected (Risk Score {int_score}/100). Navigation strictly blocked to prevent exploitation."
    elif final_action == "WARN":
        explanation = f"Warning Advisory: Elevated risk ({int_score}/100) or ambiguous threat indicators detected. Direct navigation blocked; explicit confirmation required."
    else:
        explanation = f"Proceed Permitted: Low combined risk ({int_score}/100) and clean indicators support safe navigation."

    if is_override:
        explanation += f" Security Policy Intervened: {override_reason}"

    return SecurityDecision(
        decoded_payload=clean_payload,
        normalized_url=norm_url,
        risk_score=final_risk_score,
        confidence=final_confidence,
        classification=classification,
        evidence_list=list(evidence_map.values()),
        threat_intelligence_status=ti_status,
        recommended_action=raw_action,
        final_enforcement_action=final_action,
        payload_type=ptype,
        explanation=explanation,
        is_safety_override=is_override,
        override_reason=override_reason,
        state_key=state_key,
        q_values=q_values,
        safety_rule_applied=safety_rule,
        threat_intel_details=ti_details,
        redirect_details=redirect_details,
        tls_details=tls_details,
    )


class ThreatPreventionEngine:
    """Executes adaptive threat prevention workflows based on RL and safety policies."""

    def __init__(
        self,
        rl_agent: Optional[QLearningAgent] = None,
        safety_config: Optional[SafetyConstraintsConfig] = None,
    ) -> None:
        self.rl_agent = rl_agent or get_rl_agent()
        self.safety_config = safety_config or SafetyConstraintsConfig()

    def evaluate_threat(
        self,
        raw_payload: str,
        payload_type: str,
        analysis_result: Optional[URLAnalysisResult] = None,
        ml_prediction: Optional[Any] = None,
    ) -> SecurityDecision:
        """Evaluate a scanned payload and decide whether to ALLOW, WARN, or BLOCK."""
        return evaluate_security_decision(
            decoded_payload=raw_payload,
            payload_type=payload_type,
            analysis_result=analysis_result,
            ml_prediction=ml_prediction,
            enable_threat_intel=True,
            enable_live_redirects=True,
            enable_live_tls=True,
            rl_agent=self.rl_agent,
            safety_config=self.safety_config,
        )

    def evaluate_navigation_decision(
        self,
        raw_payload: str,
        payload_type: str,
        analysis_result: Optional[URLAnalysisResult] = None,
        ml_prediction: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Centralized navigation safety decision for QR destination opening."""
        decision = evaluate_security_decision(
            decoded_payload=raw_payload,
            payload_type=payload_type,
            analysis_result=analysis_result,
            ml_prediction=ml_prediction,
            enable_threat_intel=True,
            enable_live_redirects=True,
            enable_live_tls=True,
            rl_agent=self.rl_agent,
            safety_config=self.safety_config,
        )

        clean_payload = (raw_payload or "").strip()
        try:
            parsed = urlparse(clean_payload)
            scheme = (parsed.scheme or "").lower()
            hostname = (parsed.hostname or "").lower()
            path = parsed.path or ""
        except Exception:
            scheme = ""
            hostname = ""
            path = ""

        is_web_url = scheme in ("https", "http") and bool(hostname)
        is_google_form = is_web_url and (
            "forms.gle" in hostname or ("docs.google.com" in hostname and "/forms" in path.lower())
        )

        # Retrieve risk indicators
        indicators = [
            e["description"]
            for e in decision.evidence_list
            if e.get("weight", 0.0) > 0.0 or e.get("severity") in ("CRITICAL", "HIGH", "MEDIUM")
        ]

        # CASE A: Blocked / Malicious Destination
        if decision.final_enforcement_action == "BLOCK":
            return {
                "decision": decision,
                "is_permitted": False,
                "open_url": None,
                "button_label": None,
                "status_category": "BLOCKED",
                "status_badge_color": "#ef4444",
                "display_title": "🚨 Potentially Malicious QR Code Blocked",
                "display_message": (
                    "Opening this destination has been strictly disabled to protect your device from "
                    "phishing, credential theft, malware distribution, or unauthorized redirection."
                ),
                "detected_indicators": indicators or [decision.explanation],
                "is_google_form": is_google_form,
                "disclaimer": "Zero-Trust Security Policy: High-threat indicators or confirmed malicious intelligence mandated an unconditional block.",
            }

        # CASE B: Non-Web Payloads (Plain Text, Contact Details, etc.)
        if not is_web_url:
            return {
                "decision": decision,
                "is_permitted": False,
                "open_url": None,
                "button_label": None,
                "status_category": "INFO",
                "status_badge_color": "#38bdf8",
                "display_title": "📄 Plain Text / Non-Web Payload",
                "display_message": "The decoded content is plain text or an offline format, not an approved web destination.",
                "detected_indicators": indicators,
                "is_google_form": False,
                "disclaimer": "This QR payload does not launch a web browser session.",
            }

        # CASE C: Suspicious or Uncertain
        if decision.final_enforcement_action == "WARN":
            caution_url = clean_payload if is_web_url else None
            is_unverified = decision.classification == "UNVERIFIED" or scheme == "http"
            verdict_label = "UNVERIFIED" if is_unverified else "SUSPICIOUS"
            title = (
                "⚠️ Unverified Destination: Safety Cannot Be Confidently Established"
                if is_unverified
                else "⚠️ Caution: Suspicious Threat Indicators Detected"
            )
            return {
                "decision": decision,
                "is_permitted": False,
                "open_url": None,
                "button_label": None,
                "caution_url": caution_url,
                "requires_confirmation": True,
                "status_category": "UNCERTAIN",
                "verdict": verdict_label,
                "status_badge_color": "#f59e0b",
                "display_title": title,
                "display_message": (
                    "This destination exhibits elevated risk, unencrypted transport, or ambiguous security indicators. "
                    "Automatic direct navigation has been blocked. Review the security evidence below. "
                    "You may only proceed with explicit user confirmation."
                ),
                "detected_indicators": indicators,
                "is_google_form": is_google_form,
                "disclaimer": "Static URL analysis cannot guarantee detection of every malicious destination. Conservative security policy enforced.",
            }

        # CASE D: Permitted Web Destination (Low Risk & Verified ALLOW)
        button_text = "🌐 Open Google Form" if is_google_form else "🌐 Open Website"
        return {
            "decision": decision,
            "is_permitted": True,
            "open_url": clean_payload,
            "button_label": button_text,
            "caution_url": None,
            "requires_confirmation": False,
            "status_category": "PERMITTED",
            "verdict": "SAFE",
            "status_badge_color": "#10b981",
            "display_title": "🛡️ Permitted Destination: Baseline Security Checks Passed",
            "display_message": (
                "Combined risk assessment, feature analysis, and heuristic checks support safe navigation. "
                f"Click '{button_text}' below to open the destination in a new browser tab."
            ),
            "detected_indicators": indicators,
            "is_google_form": is_google_form,
            "disclaimer": (
                "Zero-Trust Notice: A low-risk classification indicates baseline checks passed, but does not guarantee "
                "the destination is 100% safe. Always verify credentials before submitting sensitive data."
            ),
        }

    def benchmark_policies(
        self,
        test_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate Adaptive Q-Learning policy vs Fixed-Threshold baseline on identical scenarios."""
        scenarios = test_scenarios or get_default_benchmark_scenarios()
        total_scenarios = len(scenarios)

        if total_scenarios == 0:
            return {"error": "No benchmark scenarios available."}

        fixed_malicious_allowed = 0
        fixed_benign_blocked = 0
        fixed_warned = 0
        fixed_total_reward = 0.0
        fixed_correct_decisions = 0

        rl_malicious_allowed = 0
        rl_benign_blocked = 0
        rl_warned = 0
        rl_total_reward = 0.0
        rl_correct_decisions = 0
        rl_overrides = 0

        total_malicious = sum(1 for s in scenarios if s["ground_truth"] == 1)
        total_benign = sum(1 for s in scenarios if s["ground_truth"] == 0)

        reward_cfg = RewardConfig()

        for s in scenarios:
            state = s["state"]
            ground_truth = s["ground_truth"]
            risk_val = s.get("risk_score", 50.0)
            is_uncertain = "UNCERTAIN" in state

            # --- 1. Fixed-Threshold Baseline (30/70) ---
            if risk_val >= 70.0:
                fixed_act = "BLOCK"
            elif risk_val >= 30.0:
                fixed_act = "WARN"
            else:
                fixed_act = "ALLOW"

            if fixed_act == "WARN":
                fixed_warned += 1
            if ground_truth == 1 and fixed_act == "ALLOW":
                fixed_malicious_allowed += 1
            if ground_truth == 0 and fixed_act == "BLOCK":
                fixed_benign_blocked += 1
            if (ground_truth == 1 and fixed_act == "BLOCK") or (ground_truth == 0 and fixed_act == "ALLOW"):
                fixed_correct_decisions += 1

            fixed_reward = compute_reward(fixed_act, ground_truth, is_uncertain, reward_cfg)
            fixed_total_reward += fixed_reward

            # --- 2. Adaptive Q-Learning Policy with Safety Enforcer ---
            raw_rl_act = self.rl_agent.choose_action(state, training=False)
            final_rl_act = raw_rl_act

            if raw_rl_act == "ALLOW":
                if "MALICIOUS" in state or "HIGH" in state or risk_val >= 50.0:
                    final_rl_act = "BLOCK"
                    rl_overrides += 1
                elif "UNCERTAIN" in state or "MED" in state or risk_val >= 20.0:
                    final_rl_act = "WARN"
                    rl_overrides += 1

            if final_rl_act == "WARN":
                rl_warned += 1
            if ground_truth == 1 and final_rl_act == "ALLOW":
                rl_malicious_allowed += 1
            if ground_truth == 0 and final_rl_act == "BLOCK":
                rl_benign_blocked += 1
            if (ground_truth == 1 and final_rl_act == "BLOCK") or (ground_truth == 0 and final_rl_act == "ALLOW"):
                rl_correct_decisions += 1

            rl_reward = compute_reward(final_rl_act, ground_truth, is_uncertain, reward_cfg)
            rl_total_reward += rl_reward

        fixed_mal_allow_rate = round((fixed_malicious_allowed / max(1, total_malicious)) * 100, 2)
        fixed_ben_block_rate = round((fixed_benign_blocked / max(1, total_benign)) * 100, 2)
        fixed_warn_rate = round((fixed_warned / total_scenarios) * 100, 2)
        fixed_accuracy = round((fixed_correct_decisions / total_scenarios) * 100, 2)

        rl_mal_allow_rate = round((rl_malicious_allowed / max(1, total_malicious)) * 100, 2)
        rl_ben_block_rate = round((rl_benign_blocked / max(1, total_benign)) * 100, 2)
        rl_warn_rate = round((rl_warned / total_scenarios) * 100, 2)
        rl_accuracy = round((rl_correct_decisions / total_scenarios) * 100, 2)

        return {
            "dataset_summary": {
                "total_scenarios": total_scenarios,
                "benign_scenarios": total_benign,
                "malicious_scenarios": total_malicious,
            },
            "fixed_threshold_policy": {
                "name": "Fixed Risk-Threshold Policy (30/70)",
                "malicious_allow_rate": fixed_mal_allow_rate,
                "benign_block_rate": fixed_ben_block_rate,
                "warning_rate": fixed_warn_rate,
                "optimal_decision_rate": fixed_accuracy,
                "total_cumulative_reward": round(fixed_total_reward, 2),
            },
            "q_learning_adaptive_policy": {
                "name": "Q-Learning Adaptive Prevention (with Safety Enforcer)",
                "malicious_allow_rate": rl_mal_allow_rate,
                "benign_block_rate": rl_ben_block_rate,
                "warning_rate": rl_warn_rate,
                "optimal_decision_rate": rl_accuracy,
                "total_cumulative_reward": round(rl_total_reward, 2),
                "safety_overrides_applied": rl_overrides,
            },
        }

    # Alias for API compatibility
    evaluate_baseline_comparison = benchmark_policies


def get_default_benchmark_scenarios() -> List[Dict[str, Any]]:
    """Diverse benchmark scenarios for side-by-side policy evaluation."""
    return [
        {"state": "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN", "risk_score": 5.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN", "risk_score": 10.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:MEDIUM|UNC:CERTAIN", "risk_score": 25.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:MED|HEUR:LOW|UNC:CERTAIN", "risk_score": 35.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:MED|HEUR:MEDIUM|UNC:UNCERTAIN", "risk_score": 45.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:MEDIUM|UNC:UNCERTAIN", "risk_score": 28.0, "ground_truth": 0},
        {"state": "RF:UNKNOWN|RISK:LOW|HEUR:LOW|UNC:UNCERTAIN", "risk_score": 18.0, "ground_truth": 0},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN", "risk_score": 95.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN", "risk_score": 88.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:MEDIUM|UNC:CERTAIN", "risk_score": 78.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:MED|HEUR:HIGH|UNC:CERTAIN", "risk_score": 62.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:MED|HEUR:MEDIUM|UNC:UNCERTAIN", "risk_score": 55.0, "ground_truth": 1},
        {"state": "RF:BENIGN|RISK:MED|HEUR:HIGH|UNC:UNCERTAIN", "risk_score": 52.0, "ground_truth": 1},
        {"state": "RF:UNKNOWN|RISK:HIGH|HEUR:HIGH|UNC:UNCERTAIN", "risk_score": 85.0, "ground_truth": 1},
    ]


# Global prevention engine singleton
_engine_instance: Optional[ThreatPreventionEngine] = None


def get_prevention_engine() -> ThreatPreventionEngine:
    """Retrieve global threat prevention engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ThreatPreventionEngine()
    return _engine_instance
