"""Unit tests for the Reinforcement Learning Module (Tabular Q-Learning).

Tests:
- Discrete state encoding & quantization.
- Reward calculation matrix (benign, malicious, uncertainty).
- Bellman equation Q-value updates.
- Epsilon-greedy action selection.
- Simulation training loop and cumulative reward progression.
- Policy persistence (JSON export & reload).
"""

from pathlib import Path
import pytest

from modules.reinforcement_learning import (
    ACTIONS,
    QLearningAgent,
    RewardConfig,
    compute_reward,
    encode_state,
    get_default_training_scenarios,
    get_rl_agent,
)


class TestStateEncodingAndRewards:
    """Test suite for state representation and reward calculations."""

    def test_encode_state_clean_benign(self):
        """Test state encoding for a clearly legitimate, low-risk URL."""
        state = encode_state(
            rf_pred="Benign",
            rf_risk_score=12.5,
            heuristic_category="LOW",
            confidence=95.0,
            is_model_available=True,
        )
        assert state == "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN"

    def test_encode_state_high_threat_malicious(self):
        """Test state encoding for an obvious malicious URL."""
        state = encode_state(
            rf_pred="Malicious",
            rf_risk_score=92.0,
            heuristic_category="HIGH",
            confidence=98.0,
            is_model_available=True,
        )
        assert state == "RF:MALICIOUS|RISK:HIGH|HEUR:HIGH|UNC:CERTAIN"

    def test_encode_state_uncertainty_trigger(self):
        """Test that low confidence or missing model triggers UNCERTAIN flag."""
        state_low_conf = encode_state(
            rf_pred="Benign",
            rf_risk_score=45.0,
            heuristic_category="MEDIUM",
            confidence=55.0,
            is_model_available=True,
        )
        assert "UNC:UNCERTAIN" in state_low_conf

        state_no_model = encode_state(
            rf_pred="Unavailable",
            rf_risk_score=50.0,
            heuristic_category="LOW",
            confidence=0.0,
            is_model_available=False,
        )
        assert "RF:UNKNOWN" in state_no_model
        assert "UNC:UNCERTAIN" in state_no_model

    def test_reward_benign_outcomes(self):
        """Test reward values for benign URLs."""
        cfg = RewardConfig()
        # Correctly allowing benign URL
        r_allow = compute_reward("ALLOW", ground_truth_label=0, is_uncertain=False, rewards=cfg)
        assert r_allow == 15.0

        # Unnecessary warning on clear benign
        r_warn = compute_reward("WARN", ground_truth_label=0, is_uncertain=False, rewards=cfg)
        assert r_warn == -5.0

        # Severe false positive: blocking benign URL
        r_block = compute_reward("BLOCK", ground_truth_label=0, is_uncertain=False, rewards=cfg)
        assert r_block == -50.0

    def test_reward_malicious_outcomes(self):
        """Test reward values for malicious URLs."""
        cfg = RewardConfig()
        # Correctly blocking malicious threat
        r_block = compute_reward("BLOCK", ground_truth_label=1, is_uncertain=False, rewards=cfg)
        assert r_block == 25.0

        # Warning provided on malicious
        r_warn = compute_reward("WARN", ground_truth_label=1, is_uncertain=False, rewards=cfg)
        assert r_warn == 5.0

        # Catastrophic failure: allowed malicious
        r_allow = compute_reward("ALLOW", ground_truth_label=1, is_uncertain=False, rewards=cfg)
        assert r_allow == -100.0

    def test_reward_uncertain_caution(self):
        """Test that prudent warnings on uncertain states are rewarded."""
        cfg = RewardConfig()
        r_warn_unc = compute_reward("WARN", ground_truth_label=0, is_uncertain=True, rewards=cfg)
        assert r_warn_unc == 10.0


class TestQLearningAgentCore:
    """Test suite for agent Q-value updates, action selection, and training."""

    def test_q_value_update_equation(self):
        """Verify Q(s,a) update rule: Q = Q + alpha * [r + gamma * max_Q' - Q]."""
        agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9, epsilon=0.0)
        state = "TEST:STATE_A"
        next_state = "TEST:STATE_B"

        # Initialize
        agent.initialize_state(state)
        agent.initialize_state(next_state)
        agent.q_table[next_state]["ALLOW"] = 10.0
        agent.q_table[next_state]["BLOCK"] = 20.0  # max next Q is 20.0

        # Update with reward = 5.0, done = False
        # Expected target = 5.0 + 0.9 * 20.0 = 23.0
        # New Q = 0.0 + 0.5 * (23.0 - 0.0) = 11.5
        td_error = agent.update_q_value(state, "BLOCK", reward=5.0, next_state=next_state, done=False)

        assert td_error == 23.0
        assert agent.get_q_value(state, "BLOCK") == 11.5

    def test_action_selection_greedy(self):
        """Verify that greedy action selection chooses highest Q-value."""
        agent = QLearningAgent(epsilon=0.0)
        state = "RF:BENIGN|RISK:LOW|HEUR:LOW|UNC:CERTAIN"
        agent.initialize_state(state)
        agent.q_table[state]["ALLOW"] = 25.0
        agent.q_table[state]["WARN"] = 0.0
        agent.q_table[state]["BLOCK"] = -10.0

        action = agent.choose_action(state, training=False)
        assert action == "ALLOW"

    def test_action_selection_exploration(self):
        """Verify that epsilon=1.0 explores across all valid actions."""
        agent = QLearningAgent(epsilon=1.0)
        state = "TEST:STATE_EXPLORE"
        actions_chosen = set()

        for _ in range(100):
            act = agent.choose_action(state, training=True)
            actions_chosen.add(act)

        assert actions_chosen == set(ACTIONS)

    def test_train_simulation_convergence(self, tmp_path: Path):
        """Test training agent in simulation and checking reward tracking."""
        q_file = tmp_path / "test_q.json"
        meta_file = tmp_path / "test_meta.json"

        agent = QLearningAgent(
            learning_rate=0.2,
            epsilon=0.5,
            min_epsilon=0.01,
            q_table_path=q_file,
            metadata_path=meta_file,
        )

        scenarios = get_default_training_scenarios()
        results = agent.train_simulation(episodes=150, scenarios=scenarios)

        assert results["episodes"] == 150
        assert len(agent.q_table) > 0
        assert len(results["cumulative_rewards"]) == 150
        assert q_file.exists()
        assert meta_file.exists()

    def test_save_and_reload_policy(self, tmp_path: Path):
        """Test policy serialization and persistence."""
        q_file = tmp_path / "saved_q.json"
        meta_file = tmp_path / "saved_meta.json"

        agent_a = QLearningAgent(q_table_path=q_file, metadata_path=meta_file)
        agent_a.q_table["STATE_X"] = {"ALLOW": 12.0, "WARN": 1.0, "BLOCK": -4.0}
        saved = agent_a.save_policy()
        assert saved is True

        agent_b = QLearningAgent(q_table_path=q_file, metadata_path=meta_file)
        assert len(agent_b.q_table) == 0

        loaded = agent_b.load_policy()
        assert loaded is True
        assert "STATE_X" in agent_b.q_table
        assert agent_b.get_q_value("STATE_X", "ALLOW") == 12.0
