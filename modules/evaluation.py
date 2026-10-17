"""Evaluation Module for QRShield.

Provides evaluation metrics for machine learning classification (Accuracy,
Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC) and visualization utilities
using Plotly.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics and confusion matrix.

    Args:
        y_true: Ground-truth binary labels (0 for benign, 1 for malicious).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Optional predicted probabilities for the positive class (1).

    Returns:
        Dict[str, Any]: Metrics dictionary including accuracy, precision, recall,
        f1_score, confusion_matrix, and sample distribution.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob))
        except Exception:
            roc_auc = None

    total_samples = len(y_true_arr)
    num_benign = int(np.sum(y_true_arr == 0))
    num_malicious = int(np.sum(y_true_arr == 1))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "dataset_summary": {
            "total_samples": total_samples,
            "benign_samples": num_benign,
            "malicious_samples": num_malicious,
        },
    }


def plot_confusion_matrix(
    cm_dict: Dict[str, Any],
    title: str = "Random Forest Confusion Matrix",
) -> go.Figure:
    """Generate an interactive Plotly heatmap for the confusion matrix.

    Args:
        cm_dict: Dictionary containing tn, fp, fn, tp or 2x2 matrix.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    matrix = cm_dict.get("matrix", [[cm_dict.get("tn", 0), cm_dict.get("fp", 0)],
                                    [cm_dict.get("fn", 0), cm_dict.get("tp", 0)]])

    labels = ["Benign (0)", "Malicious (1)"]
    z_values = [[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]]

    annotations = []
    cell_descriptions = [
        [f"True Negative (TN)<br><b>{z_values[0][0]}</b>", f"False Positive (FP)<br><b>{z_values[0][1]}</b>"],
        [f"False Negative (FN)<br><b>{z_values[1][0]}</b>", f"True Positive (TP)<br><b>{z_values[1][1]}</b>"],
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=cell_descriptions[i][j],
                    showarrow=False,
                    font=dict(color="white", size=13),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=labels,
            y=labels,
            colorscale=[[0, "#0f172a"], [0.5, "#1e3a8a"], [1, "#0284c7"]],
            showscale=True,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        xaxis=dict(title="Predicted Class", color="#94a3b8"),
        yaxis=dict(title="Actual Ground Truth", color="#94a3b8", autorange="reversed"),
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    top_n: int = 10,
    title: str = "Top Random Forest Feature Importances",
) -> go.Figure:
    """Generate a horizontal bar chart displaying top feature importances.

    Args:
        feature_names: List of all feature names.
        importances: List of feature importance weights.
        top_n: Number of top features to display.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names = [p[0] for p in reversed(pairs)]
    values = [round(p[1], 4) for p in reversed(pairs)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, "#38bdf8"], [1, "#0284c7"]],
            ),
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        xaxis=dict(title="Gini Importance (Normalized Weight)", color="#94a3b8"),
        yaxis=dict(color="#94a3b8"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )

    return fig


def plot_cumulative_rewards(
    cumulative_rewards: List[float],
    title: str = "Q-Learning Cumulative Reward Progression",
) -> go.Figure:
    """Generate interactive Plotly line chart for RL training rewards across episodes.

    Args:
        cumulative_rewards: List of cumulative reward values over training episodes.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly line chart.
    """
    episodes = list(range(1, len(cumulative_rewards) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=cumulative_rewards,
            mode="lines",
            line=dict(color="#10b981", width=2.5),
            name="Cumulative Reward",
            hovertemplate="Episode %{x}: Reward %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(title="Training Episodes", color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(title="Cumulative Reward", color="#94a3b8", gridcolor="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=300,
    )

    return fig


def plot_action_distribution(
    action_distribution: Dict[str, Any],
    title: str = "Trained Policy Action Distribution",
) -> go.Figure:
    """Generate a styled donut/bar chart showing distribution of ALLOW, WARN, and BLOCK actions.

    Args:
        action_distribution: Dictionary mapping actions to counts or percentage info.
        title: Chart title.

    Returns:
        go.Figure: Styled Plotly pie figure.
    """
    labels = []
    values = []
    colors = []

    color_map = {
        "ALLOW": "#10b981",
        "WARN": "#f59e0b",
        "BLOCK": "#ef4444",
    }

    for act, data in action_distribution.items():
        labels.append(act)
        val = data.get("count", data) if isinstance(data, dict) else data
        values.append(val)
        colors.append(color_map.get(act, "#64748b"))

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} choices (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=50, b=30),
        height=300,
        legend=dict(font=dict(color="#cbd5e1")),
    )

    return fig


def _create_empty_figure(title: str, message: str = "No telemetry records available yet.") -> go.Figure:
    """Generate a clean dark-themed placeholder figure when dataset is empty."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"📊 {message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#94a3b8"),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def plot_threat_category_distribution(
    scans: List[Dict[str, Any]],
    title: str = "Threat Category Distribution: ML vs. Heuristic",
) -> go.Figure:
    """Generate a side-by-side grouped bar chart contrasting Random Forest classifications
    with Static Heuristic risk severity tiers.

    Args:
        scans: List of scan history dictionaries.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    if not scans:
        return _create_empty_figure(title, "No scan records available to evaluate threat categories.")

    # 1. Random Forest Predicted Class Counts
    rf_benign = sum(1 for s in scans if str(s.get("predicted_class", "")).upper() == "BENIGN")
    rf_malicious = sum(1 for s in scans if str(s.get("predicted_class", "")).upper() == "MALICIOUS")
    rf_unknown = len(scans) - (rf_benign + rf_malicious)

    # 2. Heuristic Severity Tier Counts (based on risk_score or heuristic indicators)
    heur_low = 0
    heur_med = 0
    heur_high = 0
    for s in scans:
        score = s.get("risk_score")
        if score is None:
            heur_low += 1
            continue
        try:
            val = float(score)
            if val <= 1.0:
                val = val * 100.0
            if val < 30.0:
                heur_low += 1
            elif val < 70.0:
                heur_med += 1
            else:
                heur_high += 1
        except (ValueError, TypeError):
            heur_low += 1

    categories = ["Low Risk / Benign", "Medium / Uncertain", "High Risk / Malicious"]
    rf_values = [rf_benign, rf_unknown, rf_malicious]
    heur_values = [heur_low, heur_med, heur_high]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Random Forest ML Model",
            x=categories,
            y=rf_values,
            marker_color="#0284c7",
            text=rf_values,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Static Heuristics Engine",
            x=categories,
            y=heur_values,
            marker_color="#38bdf8",
            text=heur_values,
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(title="Record Count", color="#94a3b8", gridcolor="#1e293b"),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        legend=dict(font=dict(color="#cbd5e1"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_risk_score_distribution(
    scans: List[Dict[str, Any]],
    title: str = "Telemetry Risk-Score Distribution",
) -> go.Figure:
    """Generate a binned distribution chart of risk scores with cybersecurity risk zones.

    Args:
        scans: List of scan history dictionaries.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    if not scans:
        return _create_empty_figure(title, "No scan records available to plot risk score distribution.")

    scores = []
    for s in scans:
        raw_score = s.get("risk_score")
        if raw_score is not None:
            try:
                val = float(raw_score)
                if val <= 1.0:
                    val = val * 100.0
                scores.append(round(val, 1))
            except (ValueError, TypeError):
                continue

    if not scores:
        return _create_empty_figure(title, "No numerical risk scores present in current scan records.")

    bins = ["0-19% (Safe)", "20-39% (Low)", "40-59% (Moderate)", "60-79% (Elevated)", "80-100% (Critical)"]
    counts = [0, 0, 0, 0, 0]
    colors = ["#10b981", "#34d399", "#f59e0b", "#f97316", "#ef4444"]

    for sc in scores:
        if sc < 20:
            counts[0] += 1
        elif sc < 40:
            counts[1] += 1
        elif sc < 60:
            counts[2] += 1
        elif sc < 80:
            counts[3] += 1
        else:
            counts[4] += 1

    fig = go.Figure(
        go.Bar(
            x=bins,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(title="Calculated Risk Band", color="#94a3b8"),
        yaxis=dict(title="URL Count", color="#94a3b8", gridcolor="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def plot_scan_activity_timeline(
    scans: List[Dict[str, Any]],
    title: str = "Scan Telemetry Activity Over Time",
) -> go.Figure:
    """Generate a chronological timeline chart of scan events by verdict.

    Args:
        scans: List of scan history dictionaries.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    if not scans:
        return _create_empty_figure(title, "No scan events recorded yet for timeline tracking.")

    from collections import defaultdict
    from datetime import datetime

    benign_by_date: Dict[str, int] = defaultdict(int)
    malicious_by_date: Dict[str, int] = defaultdict(int)

    for s in scans:
        ts_str = s.get("timestamp")
        if not ts_str:
            date_label = "Undated"
        else:
            try:
                # Handle ISO timestamps with or without timezone
                cleaned_ts = str(ts_str).replace("Z", "+00:00")
                dt = datetime.fromisoformat(cleaned_ts)
                date_label = dt.strftime("%Y-%m-%d")
            except Exception:
                date_label = str(ts_str)[:10]

        pred = str(s.get("predicted_class", "")).upper()
        if pred == "MALICIOUS":
            malicious_by_date[date_label] += 1
        else:
            benign_by_date[date_label] += 1

    all_dates = sorted(list(set(list(benign_by_date.keys()) + list(malicious_by_date.keys()))))
    benign_counts = [benign_by_date[d] for d in all_dates]
    malicious_counts = [malicious_by_date[d] for d in all_dates]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Benign / Low Risk",
            x=all_dates,
            y=benign_counts,
            marker_color="#10b981",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Malicious / High Risk",
            x=all_dates,
            y=malicious_counts,
            marker_color="#ef4444",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(title="Date", color="#94a3b8"),
        yaxis=dict(title="Scans Recorded", color="#94a3b8", gridcolor="#1e293b"),
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        legend=dict(font=dict(color="#cbd5e1"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_frequent_indicators(
    scans: List[Dict[str, Any]],
    top_n: int = 8,
    title: str = "Most Frequently Detected URL Indicators",
) -> go.Figure:
    """Generate a horizontal bar chart displaying the most frequent threat indicators detected.

    Args:
        scans: List of scan history dictionaries.
        top_n: Maximum number of top indicators to show.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    if not scans:
        return _create_empty_figure(title, "No scan records available to aggregate threat indicators.")

    from collections import Counter

    indicator_counts: Counter = Counter()
    for s in scans:
        indicators = s.get("detected_indicators") or []
        if isinstance(indicators, list):
            for ind in indicators:
                if ind and str(ind).strip():
                    indicator_counts[str(ind).strip()] += 1
        elif isinstance(indicators, str) and indicators.strip():
            # Handle semicolon-separated strings from CSV
            for ind in indicators.split(";"):
                if ind and ind.strip():
                    indicator_counts[ind.strip()] += 1

    if not indicator_counts:
        return _create_empty_figure(title, "No suspicious indicators triggered across recorded scans.")

    most_common = indicator_counts.most_common(top_n)
    labels = [item[0] for item in reversed(most_common)]
    counts = [item[1] for item in reversed(most_common)]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#38bdf8"], [1, "#0284c7"]],
            ),
            text=counts,
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(title="Detections Count", color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(color="#94a3b8"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def plot_prevention_action_distribution(
    scans: List[Dict[str, Any]],
    title: str = "Enforced Prevention Action Distribution",
) -> go.Figure:
    """Generate a donut chart showing the breakdown of ALLOW, WARN, and BLOCK actions.

    Args:
        scans: List of scan history dictionaries.
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    if not scans:
        return _create_empty_figure(title, "No prevention decisions recorded yet.")

    action_counts = {"ALLOW": 0, "WARN": 0, "BLOCK": 0}
    for s in scans:
        act = str(s.get("prevention_action", "")).upper()
        if "ALLOW" in act:
            action_counts["ALLOW"] += 1
        elif "WARN" in act:
            action_counts["WARN"] += 1
        elif "BLOCK" in act:
            action_counts["BLOCK"] += 1

    total = sum(action_counts.values())
    if total == 0:
        return _create_empty_figure(title, "No action decisions recorded in telemetry.")

    labels = ["ALLOW", "WARN", "BLOCK"]
    values = [action_counts["ALLOW"], action_counts["WARN"], action_counts["BLOCK"]]
    colors = ["#10b981", "#f59e0b", "#ef4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} events (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=50, b=30),
        height=300,
        legend=dict(font=dict(color="#cbd5e1")),
    )
    return fig


def plot_rl_vs_baseline_comparison(
    comparison_data: Dict[str, Any],
    title: str = "Policy Decision Performance: Fixed Baseline vs. Adaptive RL",
) -> go.Figure:
    """Generate a grouped bar chart comparing Fixed-Threshold baseline with Adaptive Q-Learning.

    Args:
        comparison_data: Dictionary returned by ThreatPreventionEngine.benchmark_policies().
        title: Plot title.

    Returns:
        go.Figure: Styled Plotly figure.
    """
    fixed = comparison_data.get("fixed_threshold_policy", {})
    rl = comparison_data.get("q_learning_adaptive_policy", {})

    metrics = [
        "Optimal Decision Rate",
        "Warning Rate",
        "Legitimate Block Rate",
        "Malicious Allow Rate",
    ]

    fixed_vals = [
        fixed.get("optimal_decision_rate", 0.0),
        fixed.get("warning_rate", 0.0),
        fixed.get("benign_block_rate", 0.0),
        fixed.get("malicious_allow_rate", 0.0),
    ]

    rl_vals = [
        rl.get("optimal_decision_rate", 0.0),
        rl.get("warning_rate", 0.0),
        rl.get("benign_block_rate", 0.0),
        rl.get("malicious_allow_rate", 0.0),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Fixed Threshold (30/70)",
            x=metrics,
            y=fixed_vals,
            marker_color="#64748b",
            text=[f"{v:.1f}%" for v in fixed_vals],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Adaptive Q-Learning",
            x=metrics,
            y=rl_vals,
            marker_color="#0284c7",
            text=[f"{v:.1f}%" for v in rl_vals],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(color="#94a3b8"),
        yaxis=dict(title="Rate Percentage (%)", color="#94a3b8", gridcolor="#1e293b", range=[0, 105]),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        legend=dict(font=dict(color="#cbd5e1"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


class ModelEvaluator:
    """Evaluates ML classifier performance and generates interactive visual reports."""

    def __init__(self) -> None:
        pass

    def evaluate_classifier(
        self,
        y_true: Any,
        y_pred: Any,
        y_prob: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute standard classification metrics."""
        return compute_classification_metrics(y_true, y_pred, y_prob)

    def get_confusion_matrix_plot(self, cm_dict: Dict[str, Any]) -> go.Figure:
        """Create Plotly heatmap for confusion matrix."""
        return plot_confusion_matrix(cm_dict)

    def get_feature_importance_plot(self, feature_names: List[str], importances: List[float]) -> go.Figure:
        """Create Plotly bar chart for feature importances."""
        return plot_feature_importance(feature_names, importances)

