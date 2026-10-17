"""Unit tests for modules/evaluation.py.

Verifies classification metrics calculation, empty-state figure handling,
and all Plotly visualization functions for ML and RL analytics.
"""

import pytest
import plotly.graph_objects as go
from modules.evaluation import (
    ModelEvaluator,
    _create_empty_figure,
    compute_classification_metrics,
    plot_action_distribution,
    plot_confusion_matrix,
    plot_cumulative_rewards,
    plot_feature_importance,
    plot_frequent_indicators,
    plot_prevention_action_distribution,
    plot_risk_score_distribution,
    plot_rl_vs_baseline_comparison,
    plot_scan_activity_timeline,
    plot_threat_category_distribution,
)


def test_compute_classification_metrics():
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["dataset_summary"]["total_samples"] == 6


def test_plot_confusion_matrix():
    cm_dict = {"tn": 15, "fp": 0, "fn": 0, "tp": 15, "matrix": [[15, 0], [0, 15]]}
    fig = plot_confusion_matrix(cm_dict)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_plot_feature_importance():
    names = ["url_length", "entropy", "num_dots"]
    weights = [0.45, 0.35, 0.20]
    fig = plot_feature_importance(names, weights)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_plot_cumulative_rewards():
    rewards = [10.0, 25.0, 50.0, 75.0]
    fig = plot_cumulative_rewards(rewards)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_plot_action_distribution():
    dist = {"ALLOW": {"count": 10}, "WARN": {"count": 4}, "BLOCK": {"count": 8}}
    fig = plot_action_distribution(dist)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "pie"


def test_empty_state_visualizations():
    empty_scans = []

    f1 = plot_threat_category_distribution(empty_scans)
    assert isinstance(f1, go.Figure)
    assert len(f1.layout.annotations) > 0

    f2 = plot_risk_score_distribution(empty_scans)
    assert isinstance(f2, go.Figure)
    assert len(f2.layout.annotations) > 0

    f3 = plot_scan_activity_timeline(empty_scans)
    assert isinstance(f3, go.Figure)
    assert len(f3.layout.annotations) > 0

    f4 = plot_frequent_indicators(empty_scans)
    assert isinstance(f4, go.Figure)
    assert len(f4.layout.annotations) > 0

    f5 = plot_prevention_action_distribution(empty_scans)
    assert isinstance(f5, go.Figure)
    assert len(f5.layout.annotations) > 0


def test_populated_visualizations():
    scans = [
        {
            "scan_id": "s1",
            "timestamp": "2026-10-17T12:00:00Z",
            "extracted_url": "https://google.com",
            "predicted_class": "BENIGN",
            "risk_score": 0.12,
            "detected_indicators": ["IP in hostname", "High entropy"],
            "prevention_action": "ALLOW",
        },
        {
            "scan_id": "s2",
            "timestamp": "2026-10-17T13:00:00Z",
            "extracted_url": "http://192.168.1.1/login.php",
            "predicted_class": "MALICIOUS",
            "risk_score": 85.0,
            "detected_indicators": ["IP in hostname", "Suspicious TLD", "High entropy"],
            "prevention_action": "BLOCK",
        },
        {
            "scan_id": "s3",
            "timestamp": "2026-10-17T14:00:00Z",
            "extracted_url": "https://suspicious-bank-login.xyz",
            "predicted_class": "UNKNOWN",
            "risk_score": 45.0,
            "detected_indicators": "Suspicious TLD; Keyword trigger",
            "prevention_action": "WARN",
        },
    ]

    f1 = plot_threat_category_distribution(scans)
    assert isinstance(f1, go.Figure)
    assert len(f1.data) == 2  # RF trace and Heuristics trace

    f2 = plot_risk_score_distribution(scans)
    assert isinstance(f2, go.Figure)
    assert len(f2.data) == 1

    f3 = plot_scan_activity_timeline(scans)
    assert isinstance(f3, go.Figure)
    assert len(f3.data) == 2  # Benign and Malicious bars

    f4 = plot_frequent_indicators(scans)
    assert isinstance(f4, go.Figure)
    assert len(f4.data) == 1

    f5 = plot_prevention_action_distribution(scans)
    assert isinstance(f5, go.Figure)
    assert len(f5.data) == 1
    assert f5.data[0].type == "pie"


def test_plot_rl_vs_baseline_comparison():
    data = {
        "fixed_threshold_policy": {
            "name": "Fixed Policy",
            "malicious_allow_rate": 0.0,
            "benign_block_rate": 0.0,
            "warning_rate": 35.7,
            "optimal_decision_rate": 64.3,
            "total_cumulative_reward": 120.0,
        },
        "q_learning_adaptive_policy": {
            "name": "Adaptive Policy",
            "malicious_allow_rate": 0.0,
            "benign_block_rate": 0.0,
            "warning_rate": 28.5,
            "optimal_decision_rate": 71.4,
            "total_cumulative_reward": 145.0,
        },
    }
    fig = plot_rl_vs_baseline_comparison(data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Fixed vs RL


def test_model_evaluator_class():
    evaluator = ModelEvaluator()
    m = evaluator.evaluate_classifier([0, 1], [0, 1])
    assert m["accuracy"] == 1.0

    cm_fig = evaluator.get_confusion_matrix_plot(m["confusion_matrix"])
    assert isinstance(cm_fig, go.Figure)

    fi_fig = evaluator.get_feature_importance_plot(["f1", "f2"], [0.6, 0.4])
    assert isinstance(fi_fig, go.Figure)
