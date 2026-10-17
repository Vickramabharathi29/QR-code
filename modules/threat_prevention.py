"""Threat Prevention Module for QRShield.

Translates reinforcement learning decisions into concrete threat prevention actions,
applies deterministic security constraints, and provides baseline comparisons.

Safety Constraints:
- Malformed or unsupported URLs are unconditionally BLOCKED.
- High-risk URLs (heuristic HIGH or ML risk >= 70) are NEVER allowed, even if RL recommends ALLOW.
- Model uncertainty triggers conservative fallback (WARN or BLOCK).
- Zero remote network calls: static safety evaluation only.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import config
from modules.reinforcement_learning import (
    ACTIONS,
    QLearningAgent,
    RewardConfig,
    compute_reward,
    encode_state,
    get_rl_agent,
)
from modules.url_analyzer import URLAnalysisResult

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
    high_risk_threshold: float = 70.0
    medium_risk_threshold: float = 30.0


@dataclass
class PreventionDecision:
    """Outcome of adaptive threat prevention evaluation."""

    raw_action: str             # Q-learning suggested action ("ALLOW", "WARN", "BLOCK")
    final_action: str           # Security-enforced action ("ALLOW", "WARN", "BLOCK")
    state_key: str              # Discrete state string
    is_safety_override: bool    # True if safety constraint modified raw_action
    override_reason: Optional[str]
    explanation: str            # Human-readable decision rationale
    q_values: Dict[str, float]  # Q-values for candidate actions
    safety_rule_applied: Optional[str]
    risk_score: float
    predicted_class: str
    heuristic_category: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to serializable dictionary."""
        return asdict(self)


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
        analysis_result: URLAnalysisResult,
        ml_prediction: Optional[Any] = None,
    ) -> PreventionDecision:
        """Evaluate a scanned payload and decide whether to ALLOW, WARN, or BLOCK.

        Enforces deterministic safety constraints over RL recommendations.

        Args:
            raw_payload: The raw text or URL embedded in the QR code.
            payload_type: Payload type classification from QR decoder.
            analysis_result: Static lexical/heuristic analysis result.
            ml_prediction: Optional MLPredictionResult from Random Forest classifier.

        Returns:
            PreventionDecision: The raw recommendation, final enforced action, and reasoning.
        """
        # 1. Non-standard or dangerous schemes handling
        is_malformed = (
            "MALFORMED" in analysis_result.status
            or analysis_result.status in ["MISSING_HOSTNAME", "INVALID_PORT"]
            or (payload_type in ["URL_HTTPS", "URL_HTTP"] and not analysis_result.is_valid_url)
        )

        if payload_type == "DANGEROUS_SCHEME" or is_malformed:
            return PreventionDecision(
                raw_action="BLOCK",
                final_action="BLOCK",
                state_key="PAYLOAD:DANGEROUS_OR_MALFORMED",
                is_safety_override=True,
                override_reason="Critical Safety Constraint: Payload contains a dangerous URI scheme or malformed structure.",
                explanation=(
                    "Navigation is blocked unconditionally. The QR payload contains a dangerous protocol "
                    "(e.g., javascript:, data:, file:) or syntax error that presents immediate exploit hazard."
                ),
                q_values={"ALLOW": -100.0, "WARN": -50.0, "BLOCK": 25.0},
                safety_rule_applied="CRITICAL_DANGEROUS_OR_MALFORMED",
                risk_score=100.0,
                predicted_class="Malicious",
                heuristic_category="HIGH",
            )

        if payload_type == "PLAIN_TEXT":
            return PreventionDecision(
                raw_action="ALLOW",
                final_action="ALLOW",
                state_key="PAYLOAD:PLAIN_TEXT",
                is_safety_override=False,
                override_reason=None,
                explanation="Safe to view. The QR payload is plain text without active protocol execution.",
                q_values={"ALLOW": 15.0, "WARN": -5.0, "BLOCK": -50.0},
                safety_rule_applied=None,
                risk_score=0.0,
                predicted_class="Benign",
                heuristic_category="LOW",
            )

        if payload_type == "SPECIALIZED_SCHEME":
            return PreventionDecision(
                raw_action="WARN",
                final_action="WARN",
                state_key="PAYLOAD:SPECIALIZED_SCHEME",
                is_safety_override=True,
                override_reason="Safety Constraint: Specialized URI protocols (tel:, mailto:, wifi:) require explicit user confirmation.",
                explanation=(
                    "Caution required: Specialized device action detected. Requires user acknowledgement "
                    "before launching native application handlers."
                ),
                q_values={"ALLOW": -10.0, "WARN": 10.0, "BLOCK": -20.0},
                safety_rule_applied="RULE_SPECIALIZED_PROTOCOL",
                risk_score=35.0,
                predicted_class="Benign",
                heuristic_category="MEDIUM",
            )

        # 2. Extract inputs for standard HTTP/HTTPS URLs
        rf_pred = "Unknown"
        rf_risk = analysis_result.risk_score * 100.0
        rf_conf = 50.0
        is_model_available = False

        if ml_prediction is not None and getattr(ml_prediction, "is_model_available", False):
            rf_pred = ml_prediction.predicted_label
            rf_risk = ml_prediction.risk_score
            rf_conf = ml_prediction.confidence
            is_model_available = True

        heur_cat = analysis_result.risk_category

        # 3. Discretize into RL State Key
        state_key = encode_state(
            rf_pred=rf_pred,
            rf_risk_score=rf_risk,
            heuristic_category=heur_cat,
            confidence=rf_conf,
            is_model_available=is_model_available,
        )

        # 4. Q-Learning action recommendation
        raw_action = self.rl_agent.choose_action(state_key, training=False)
        q_values = self.rl_agent.get_q_values(state_key)

        # 5. Apply Security Constraints & Safety Policy
        final_action = raw_action
        is_override = False
        override_reason = None
        safety_rule = None

        if raw_action == "ALLOW":
            # Safety Rule A: If ML predicted Malicious, never ALLOW
            if self.safety_config.prevent_allow_on_malicious_ml and rf_pred == "Malicious":
                final_action = "BLOCK"
                is_override = True
                override_reason = "Safety Override: Q-Learning suggested ALLOW, but Random Forest identified URL as Malicious."
                safety_rule = "SAFETY_OVERRIDE_MALICIOUS_ML"

            # Safety Rule B: If High Risk score or High Heuristic threat, never ALLOW
            elif self.safety_config.prevent_allow_on_high_risk and (rf_risk >= self.safety_config.high_risk_threshold or heur_cat == "HIGH"):
                final_action = "BLOCK"
                is_override = True
                override_reason = f"Safety Override: High threat score ({rf_risk:.1f}/100) or heuristic HIGH mandates BLOCK."
                safety_rule = "SAFETY_OVERRIDE_HIGH_THREAT"

            # Safety Rule C: If uncertain with medium heuristic risk, downgrade to WARN
            elif self.safety_config.prevent_allow_on_uncertain_medium and "UNCERTAIN" in state_key and heur_cat == "MEDIUM":
                final_action = "WARN"
                is_override = True
                override_reason = "Safety Prudence: Downgraded ALLOW to WARN due to model uncertainty and medium heuristic risk."
                safety_rule = "SAFETY_PRUDENCE_UNCERTAIN"

        # 6. Formulate Decision Explanation
        explanation_parts = []
        if is_override:
            explanation_parts.append(f"Security Policy Intervened: {override_reason}")
            explanation_parts.append(
                f"The unconstrained Q-table recommended '{raw_action}', but QRShield safety policy enforced '{final_action}'."
            )
        else:
            if final_action == "ALLOW":
                explanation_parts.append(
                    f"Proceed Permitted: Low combined risk ({rf_risk:.1f}/100) and clean heuristic indicators support safe navigation."
                )
            elif final_action == "WARN":
                explanation_parts.append(
                    f"Warning Advisory: Elevated risk ({rf_risk:.1f}/100) or ambiguous feature signals require explicit user confirmation."
                )
            elif final_action == "BLOCK":
                explanation_parts.append(
                    f"Threat Neutralized: Malicious patterns detected (Risk {rf_risk:.1f}/100, Heuristic {heur_cat}). Navigation strictly blocked."
                )

        explanation = " ".join(explanation_parts)

        return PreventionDecision(
            raw_action=raw_action,
            final_action=final_action,
            state_key=state_key,
            is_safety_override=is_override,
            override_reason=override_reason,
            explanation=explanation,
            q_values=q_values,
            safety_rule_applied=safety_rule,
            risk_score=rf_risk,
            predicted_class=rf_pred,
            heuristic_category=heur_cat,
        )

    def benchmark_policies(
        self,
        test_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate Adaptive Q-Learning policy vs Fixed-Threshold baseline on identical scenarios.

        Args:
            test_scenarios: List of benchmark evaluation scenarios.

        Returns:
            Dict[str, Any]: Side-by-side comparative performance metrics.
        """
        scenarios = test_scenarios or get_default_benchmark_scenarios()
        total_scenarios = len(scenarios)

        if total_scenarios == 0:
            return {"error": "No benchmark scenarios available."}

        # Counters for Fixed-Threshold Policy
        # Rule: risk >= 70 -> BLOCK, 30 <= risk < 70 -> WARN, risk < 30 -> ALLOW
        fixed_malicious_allowed = 0
        fixed_benign_blocked = 0
        fixed_warned = 0
        fixed_total_reward = 0.0
        fixed_correct_decisions = 0

        # Counters for Adaptive Q-Learning Policy (with Safety Constraints)
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
            ground_truth = s["ground_truth"]  # 0: Benign, 1: Malicious
            risk_val = s.get("risk_score", 50.0)
            is_uncertain = "UNCERTAIN" in state

            # --- 1. Fixed-Threshold Baseline ---
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

            # Fixed policy correctness: malicious stopped (BLOCK/WARN) and benign permitted (ALLOW)
            if (ground_truth == 1 and fixed_act == "BLOCK") or (ground_truth == 0 and fixed_act == "ALLOW"):
                fixed_correct_decisions += 1

            fixed_reward = compute_reward(fixed_act, ground_truth, is_uncertain, reward_cfg)
            fixed_total_reward += fixed_reward

            # --- 2. Adaptive Q-Learning Policy ---
            raw_rl_act = self.rl_agent.choose_action(state, training=False)
            final_rl_act = raw_rl_act

            # Apply safety constraints
            if raw_rl_act == "ALLOW":
                if "MALICIOUS" in state or "HIGH" in state:
                    final_rl_act = "BLOCK"
                    rl_overrides += 1
                elif "UNCERTAIN" in state and "MED" in state:
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

        # Compute percentages
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
        {"state": "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN", "risk_score": 10.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN", "risk_score": 15.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:MEDIUM|UNC:CERTAIN", "risk_score": 25.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:MED|HEUR:LOW|UNC:CERTAIN", "risk_score": 35.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:MED|HEUR:MEDIUM|UNC:UNCERTAIN", "risk_score": 50.0, "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:MEDIUM|UNC:UNCERTAIN", "risk_score": 28.0, "ground_truth": 0},
        {"state": "RF:UNKNOWN|RISK:LOW|HEUR:LOW|UNC:UNCERTAIN", "risk_score": 20.0, "ground_truth": 0},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN", "risk_score": 95.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN", "risk_score": 88.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:MEDIUM|UNC:CERTAIN", "risk_score": 78.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:MED|HEUR:HIGH|UNC:CERTAIN", "risk_score": 62.0, "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:MED|HEUR:MEDIUM|UNC:UNCERTAIN", "risk_score": 55.0, "ground_truth": 1},
        {"state": "RF:BENIGN|RISK:MED|HEUR:HIGH|UNC:UNCERTAIN", "risk_score": 45.0, "ground_truth": 1},  # Deceptive phishing
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
