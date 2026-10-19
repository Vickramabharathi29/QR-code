"""Reinforcement Learning Module for QRShield.

Implements tabular Q-Learning for adaptive threat prevention.
States are discretely quantized from Random Forest predictions, risk score bins,
heuristic risk categories, and confidence/uncertainty indicators.
Actions: ALLOW, WARN, BLOCK.

Update rule:
    Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import config

logger = logging.getLogger("qrshield.rl")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Supported actions
ACTIONS = ["ALLOW", "WARN", "BLOCK"]

# Discrete quantization labels
RF_CATEGORIES = ["BENIGN", "MALICIOUS", "UNKNOWN"]
RISK_BINS = ["LOW", "MED", "HIGH"]
HEURISTIC_CATEGORIES = ["LOW", "MEDIUM", "HIGH"]
UNCERTAINTY_FLAGS = ["CERTAIN", "UNCERTAIN"]

DEFAULT_Q_TABLE_PATH = config.MODELS_DIR / "q_table.json"
DEFAULT_RL_METADATA_PATH = config.MODELS_DIR / "rl_metadata.json"


@dataclass
class RewardConfig:
    """Configurable reward structure for Q-Learning agent."""

    # Benign URL outcomes
    reward_allow_benign: float = 15.0        # Optimal: allowed legitimate URL
    penalty_warn_benign: float = -5.0        # Slight user friction / alert fatigue
    penalty_block_benign: float = -50.0      # False positive: user blocked from safe URL

    # Malicious URL outcomes
    reward_block_malicious: float = 25.0     # Optimal: malicious threat intercepted
    reward_warn_malicious: float = 5.0       # Acceptable: caution provided, but not blocked
    penalty_allow_malicious: float = -100.0  # Catastrophic: malware/phishing allowed

    # Uncertainty handling
    reward_uncertain_warn: float = 10.0      # Prudent caution under high uncertainty
    penalty_uncertain_allow: float = -20.0   # Reckless allowance under uncertainty

    def to_dict(self) -> Dict[str, float]:
        """Serialize reward configuration."""
        return asdict(self)


def encode_state(
    rf_pred: str,
    rf_risk_score: float,
    heuristic_category: str,
    confidence: float = 100.0,
    is_model_available: bool = True,
) -> str:
    """Discretize continuous signals into a manageable tabular state string.

    Args:
        rf_pred: Predicted class ("Benign", "Malicious", or "Unavailable").
        rf_risk_score: Model-derived risk score (0.0 to 100.0).
        heuristic_category: Rule-based category ("LOW", "MEDIUM", "HIGH").
        confidence: Prediction confidence percentage (0.0 to 100.0).
        is_model_available: Flag indicating whether RF model is operational.

    Returns:
        str: Standardized discrete state key (e.g. 'RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN').
    """
    # 1. RF Category
    rf_norm = str(rf_pred).strip().upper()
    if not is_model_available or rf_norm in ["UNAVAILABLE", "ERROR", "UNKNOWN"]:
        rf_cat = "UNKNOWN"
    elif "MALICIOUS" in rf_norm:
        rf_cat = "MALICIOUS"
    else:
        rf_cat = "BENIGN"

    # 2. Risk Score Bin
    # Aligned with canonical thresholds: 0-19 LOW, 20-49 MED, >=50 HIGH/CRITICAL
    score = float(rf_risk_score) if rf_risk_score is not None else 50.0
    if score < 20.0:
        risk_bin = "LOW"
    elif score < 50.0:
        risk_bin = "MED"
    else:
        risk_bin = "HIGH"

    # 3. Heuristic Category
    heur_norm = str(heuristic_category).strip().upper()
    if "CRITICAL" in heur_norm or "HIGH" in heur_norm:
        heur_cat = "HIGH"
    elif "MED" in heur_norm:
        heur_cat = "MEDIUM"
    else:
        heur_cat = "LOW"

    # 4. Uncertainty Indicator
    # Considered uncertain if confidence is low (<65%), model is missing, or signals strongly conflict
    conf = float(confidence) if confidence is not None else 50.0
    conflict = (rf_cat == "BENIGN" and heur_cat == "HIGH") or (rf_cat == "MALICIOUS" and heur_cat == "LOW")
    if not is_model_available or conf < 65.0 or conflict:
        uncertainty = "UNCERTAIN"
    else:
        uncertainty = "CERTAIN"

    return f"RF:{rf_cat}|RISK:{risk_bin}|HEUR:{heur_cat}|UNC:{uncertainty}"


def compute_reward(
    action: str,
    ground_truth_label: Union[int, str],
    is_uncertain: bool = False,
    rewards: Optional[RewardConfig] = None,
) -> float:
    """Calculate reward based on action taken against verified ground truth.

    Args:
        action: Selected action ("ALLOW", "WARN", "BLOCK").
        ground_truth_label: 0 / 'BENIGN' for safe, 1 / 'MALICIOUS' for malicious.
        is_uncertain: Whether state has the uncertainty indicator.
        rewards: Optional reward configuration object.

    Returns:
        float: Computed reward value.
    """
    cfg = rewards or RewardConfig()
    is_malicious = False
    if isinstance(ground_truth_label, int):
        is_malicious = (ground_truth_label == 1)
    else:
        is_malicious = ("MALICIOUS" in str(ground_truth_label).upper())

    act = str(action).strip().upper()

    if not is_malicious:
        # Ground Truth: BENIGN
        if act == "ALLOW":
            return cfg.reward_allow_benign
        elif act == "WARN":
            return cfg.reward_uncertain_warn if is_uncertain else cfg.penalty_warn_benign
        elif act == "BLOCK":
            return cfg.penalty_block_benign
    else:
        # Ground Truth: MALICIOUS
        if act == "BLOCK":
            return cfg.reward_block_malicious
        elif act == "WARN":
            return cfg.reward_warn_malicious
        elif act == "ALLOW":
            return cfg.penalty_allow_malicious

    return 0.0


class QLearningAgent:
    """Tabular Q-Learning agent for adaptive threat prevention policies."""

    def __init__(
        self,
        actions: Optional[List[str]] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.90,
        epsilon: float = 0.20,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.995,
        q_table_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.actions = actions or ACTIONS
        self.lr = float(learning_rate)
        self.gamma = float(discount_factor)
        self.epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.q_table_path = Path(q_table_path) if q_table_path else DEFAULT_Q_TABLE_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_RL_METADATA_PATH
        self.rewards = reward_config or RewardConfig()

        # Q-table: mapping state_key -> {action: q_value}
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.training_metadata: Dict[str, Any] = {}

    def get_state_key(
        self,
        rf_pred: str,
        rf_risk_score: float,
        heuristic_category: str,
        confidence: float = 100.0,
        is_model_available: bool = True,
    ) -> str:
        """Helper to compute state key."""
        return encode_state(rf_pred, rf_risk_score, heuristic_category, confidence, is_model_available)

    def initialize_state(self, state: str) -> None:
        """Initialize Q-values for a state if not already present."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

    def get_q_value(self, state: str, action: str) -> float:
        """Retrieve Q-value for a state-action pair."""
        self.initialize_state(state)
        return self.q_table[state].get(action, 0.0)

    def get_q_values(self, state: str) -> Dict[str, float]:
        """Get copy of all action Q-values for a given state."""
        self.initialize_state(state)
        return {a: round(v, 4) for a, v in self.q_table[state].items()}

    def choose_action(self, state: str, training: bool = False) -> str:
        """Select action using epsilon-greedy policy.

        Args:
            state: Discrete state representation.
            training: If True, explores with probability epsilon.

        Returns:
            str: Selected action ('ALLOW', 'WARN', or 'BLOCK').
        """
        self.initialize_state(state)

        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation: select action with highest Q-value
        q_vals = self.q_table[state]
        max_q = max(q_vals.values())

        # Collect all actions with highest Q-value (tie handling)
        best_actions = [act for act, val in q_vals.items() if abs(val - max_q) < 1e-9]

        if len(best_actions) == 1:
            return best_actions[0]

        # Deterministic sensible tie-breaker if all Q-values are equal
        if "HIGH" in state or "MALICIOUS" in state:
            return "BLOCK" if "BLOCK" in best_actions else best_actions[0]
        elif "LOW" in state and "BENIGN" in state and "CERTAIN" in state:
            return "ALLOW" if "ALLOW" in best_actions else best_actions[0]
        else:
            return "WARN" if "WARN" in best_actions else best_actions[0]

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str] = None,
        done: bool = True,
    ) -> float:
        """Update Q-value using standard Bellman equation:

        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]

        Args:
            state: Current state key.
            action: Action taken.
            reward: Received numerical reward.
            next_state: Resulting state (or None if terminal).
            done: Whether episode terminates after this transition.

        Returns:
            float: Temporal difference error (TD error).
        """
        self.initialize_state(state)

        current_q = self.q_table[state][action]

        if done or next_state is None:
            target = reward
        else:
            self.initialize_state(next_state)
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q

        td_error = target - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[state][action] = new_q

        return td_error

    def is_policy_loaded(self) -> bool:
        """Check whether Q-table contains trained or loaded states."""
        return len(self.q_table) > 0

    def save_policy(
        self,
        q_table_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Save Q-table and training metadata to disk.

        Args:
            q_table_path: Destination JSON file for Q-table.
            metadata_path: Destination JSON file for training metadata.

        Returns:
            bool: True if saving succeeded.
        """
        q_path = Path(q_table_path) if q_table_path else self.q_table_path
        m_path = Path(metadata_path) if metadata_path else self.metadata_path

        try:
            q_path.parent.mkdir(parents=True, exist_ok=True)
            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(self.q_table, f, indent=2)

            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(self.training_metadata, f, indent=2)

            logger.info(f"Saved Q-table ({len(self.q_table)} states) to {q_path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to save Q-table to {q_path}: {exc}")
            return False

    def load_policy(
        self,
        q_table_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Load Q-table and metadata from disk.

        Args:
            q_table_path: Path to Q-table JSON file.
            metadata_path: Path to metadata JSON file.

        Returns:
            bool: True if successfully loaded.
        """
        q_path = Path(q_table_path) if q_table_path else self.q_table_path
        m_path = Path(metadata_path) if metadata_path else self.metadata_path

        if not q_path.exists():
            logger.warning(f"Q-table file not found at {q_path}.")
            return False

        try:
            with open(q_path, "r", encoding="utf-8") as f:
                loaded_table = json.load(f)

            if isinstance(loaded_table, dict):
                self.q_table = loaded_table
                logger.info(f"Loaded Q-table ({len(self.q_table)} states) from {q_path}")

            if m_path.exists():
                with open(m_path, "r", encoding="utf-8") as f:
                    self.training_metadata = json.load(f)

            return True
        except Exception as exc:
            logger.error(f"Failed to load Q-table from {q_path}: {exc}")
            return False

    def train_simulation(
        self,
        episodes: int = 500,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Train Q-Learning agent over labelled simulation scenarios.

        Uses verified scenario ground truth (never unverified user feedback or
        treating RF model prediction as unquestionable truth).

        Args:
            episodes: Total training episodes.
            scenarios: List of scenario dictionaries with state features and ground_truth.
            random_seed: Seed for reproducible simulation.

        Returns:
            Dict[str, Any]: Comprehensive training summary, reward progression, and action distribution.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Default synthetic benchmark scenarios if none provided
        training_scenarios = scenarios or get_default_training_scenarios()
        if not training_scenarios:
            raise ValueError("No training scenarios provided for Q-Learning simulation.")

        reward_history: List[float] = []
        cumulative_rewards: List[float] = []
        action_counts = {"ALLOW": 0, "WARN": 0, "BLOCK": 0}
        total_cum_reward = 0.0

        current_epsilon = self.epsilon

        for ep in range(episodes):
            scenario = random.choice(training_scenarios)
            state = scenario["state"]
            ground_truth = scenario["ground_truth"]  # 0: Benign, 1: Malicious
            is_uncertain = "UNCERTAIN" in state

            # Epsilon-greedy action choice
            if random.random() < current_epsilon:
                action = random.choice(self.actions)
            else:
                action = self.choose_action(state, training=False)

            action_counts[action] += 1

            # Compute reward against ground truth
            reward = compute_reward(action, ground_truth, is_uncertain, self.rewards)

            # Update Q-table (single-step episodic contextual update)
            self.update_q_value(state, action, reward, next_state=None, done=True)

            total_cum_reward += reward
            reward_history.append(round(reward, 2))
            cumulative_rewards.append(round(total_cum_reward, 2))

            # Decay exploration rate
            current_epsilon = max(self.min_epsilon, current_epsilon * self.epsilon_decay)

        # Assemble training metadata
        summary = {
            "algorithm": "Tabular Q-Learning",
            "episodes": episodes,
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": {
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "initial_epsilon": self.epsilon,
                "final_epsilon": round(current_epsilon, 4),
                "min_epsilon": self.min_epsilon,
            },
            "reward_config": self.rewards.to_dict(),
            "results": {
                "total_cumulative_reward": round(total_cum_reward, 2),
                "average_reward_per_episode": round(total_cum_reward / max(1, episodes), 4),
                "action_distribution": {
                    act: {
                        "count": count,
                        "percentage": round((count / max(1, episodes)) * 100, 2),
                    }
                    for act, count in action_counts.items()
                },
                "total_learned_states": len(self.q_table),
            },
            "reward_history_sample": reward_history[-100:],  # last 100 episodes
            "cumulative_rewards": cumulative_rewards,
        }

        self.training_metadata = summary
        self.save_policy()

        logger.info(
            f"Finished Q-Learning training ({episodes} episodes). "
            f"Cumulative Reward: {total_cum_reward:.2f}, States learned: {len(self.q_table)}"
        )
        return summary


def get_default_training_scenarios() -> List[Dict[str, Any]]:
    """Generate diverse, balanced scenarios with simulated ground truth for development training."""
    scenarios = [
        # Clear Benign (Label 0)
        {"state": "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN", "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN", "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:MEDIUM|UNC:CERTAIN", "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:MED|HEUR:LOW|UNC:CERTAIN", "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:MED|HEUR:MEDIUM|UNC:CERTAIN", "ground_truth": 0},

        # Ambiguous / Uncertain Benign (Label 0)
        {"state": "RF:BENIGN|RISK:MED|HEUR:MEDIUM|UNC:UNCERTAIN", "ground_truth": 0},
        {"state": "RF:BENIGN|RISK:LOW|HEUR:MEDIUM|UNC:UNCERTAIN", "ground_truth": 0},
        {"state": "RF:UNKNOWN|RISK:LOW|HEUR:LOW|UNC:UNCERTAIN", "ground_truth": 0},
        {"state": "RF:UNKNOWN|RISK:MED|HEUR:LOW|UNC:UNCERTAIN", "ground_truth": 0},

        # Clear Malicious (Label 1)
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN", "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN", "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:MEDIUM|UNC:CERTAIN", "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:MED|HEUR:HIGH|UNC:CERTAIN", "ground_truth": 1},

        # Ambiguous / Uncertain Malicious (Label 1)
        {"state": "RF:MALICIOUS|RISK:MED|HEUR:MEDIUM|UNC:UNCERTAIN", "ground_truth": 1},
        {"state": "RF:MALICIOUS|RISK:HIGH|HEUR:LOW|UNC:UNCERTAIN", "ground_truth": 1},
        {"state": "RF:BENIGN|RISK:MED|HEUR:HIGH|UNC:UNCERTAIN", "ground_truth": 1},  # Sophisticated evasion
        {"state": "RF:UNKNOWN|RISK:HIGH|HEUR:HIGH|UNC:UNCERTAIN", "ground_truth": 1},
        {"state": "RF:UNKNOWN|RISK:MED|HEUR:HIGH|UNC:UNCERTAIN", "ground_truth": 1},
    ]
    return scenarios


# Global singleton instance
_rl_agent_instance = QLearningAgent()


def get_rl_agent() -> QLearningAgent:
    """Retrieve or initialize the global Q-Learning agent singleton."""
    if not _rl_agent_instance.is_policy_loaded():
        _rl_agent_instance.load_policy()
        # If no policy file exists, train initial baseline simulation
        if not _rl_agent_instance.is_policy_loaded():
            logger.info("No saved Q-table found. Training initial baseline Q-Learning policy...")
            _rl_agent_instance.train_simulation(episodes=400)
    return _rl_agent_instance
