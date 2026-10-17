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
