"""Unit tests for the Machine Learning Classifier module (Random Forest).

Tests:
- Model training pipeline with dataset handling and train/test separation.
- Model serialization, saving, and loading from disk.
- Predictions on benign and malicious URLs.
- Predictions on feature dictionaries and raw feature vectors.
- Error handling for invalid feature vector sizes and unsupported types.
- Graceful handling when model files are missing or paths are invalid.
- Evaluation metrics and confusion matrix calculations.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from modules.feature_extractor import FEATURE_NAMES, extract_url_features
from modules.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
)
from modules.ml_classifier import (
    MaliciousURLClassifier,
    MLPredictionResult,
    get_classifier_status,
    predict_url,
)


class TestModelTrainingAndPersistence:
    """Test suite for Random Forest training, saving, and loading."""

    def test_train_and_save_artifacts(self, tmp_path: Path):
        """Test training on sample dataset and writing model + metadata."""
        custom_model_path = tmp_path / "test_model.pkl"
        custom_meta_path = tmp_path / "test_metadata.json"

        classifier = MaliciousURLClassifier(
            model_path=custom_model_path,
            metadata_path=custom_meta_path,
        )

        assert not classifier.is_trained_model_available()

        # Train on the project's dataset
        meta = classifier.train(n_estimators=10, max_depth=5, random_state=42)

        assert custom_model_path.exists()
        assert custom_meta_path.exists()
        assert classifier.is_trained_model_available()
        assert meta["model_type"] == "RandomForestClassifier"
        assert meta["feature_names"] == FEATURE_NAMES
        assert "evaluation_metrics" in meta
        assert "accuracy" in meta["evaluation_metrics"]

    def test_load_saved_model(self, tmp_path: Path):
        """Test loading an existing trained model and metadata."""
        model_path = tmp_path / "rf_model.pkl"
        meta_path = tmp_path / "meta.json"

        classifier = MaliciousURLClassifier(model_path=model_path, metadata_path=meta_path)
        classifier.train(n_estimators=10, max_depth=4)

        # Create a new classifier instance pointing to the same files
        new_classifier = MaliciousURLClassifier(model_path=model_path, metadata_path=meta_path)
        assert new_classifier.model is None

        loaded = new_classifier.load_model()
        assert loaded is True
        assert new_classifier.model is not None
        assert new_classifier.metadata["model_type"] == "RandomForestClassifier"


@pytest.fixture(scope="module")
def default_classifier():
    """Ensure standard model is loaded for testing."""
    clf = MaliciousURLClassifier()
    clf.load_model()
    return clf


class TestClassifierPredictions:
    """Test suite for classifier inferences and outputs."""

    def test_predict_benign_url(self, default_classifier: MaliciousURLClassifier):
        """Test prediction on a well-known legitimate URL."""
        url = "https://www.google.com"
        result = default_classifier.predict(url)

        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is True
        assert result.predicted_label in ["Benign", "Malicious"]
        assert 0.0 <= result.risk_score <= 100.0
        assert 50.0 <= result.confidence <= 100.0
        assert "Benign" in result.probabilities
        assert "Malicious" in result.probabilities
        assert len(result.top_contributing_features) > 0
        assert result.limitations_notice is not None

    def test_predict_malicious_url(self, default_classifier: MaliciousURLClassifier):
        """Test prediction on an obvious synthetic phishing/scam URL."""
        url = "http://paypal-security-update-verification.com/login/verify.php?token=123"
        result = default_classifier.predict(url)

        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is True
        assert result.predicted_label == "Malicious"
        assert result.risk_score >= 50.0
        assert result.predicted_class == 1

    def test_predict_from_feature_dictionary(self, default_classifier: MaliciousURLClassifier):
        """Test prediction when input is a pre-extracted feature dictionary."""
        url = "https://en.wikipedia.org/wiki/Machine_learning"
        features = extract_url_features(url)

        result = default_classifier.predict(features)
        assert isinstance(result, MLPredictionResult)
        assert result.predicted_label == "Benign"

    def test_predict_from_feature_vector(self, default_classifier: MaliciousURLClassifier):
        """Test prediction when input is a 15-float numerical vector."""
        vec = [30.0, 15.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 5.0, 0.0, 0.0, 0.0]
        result = default_classifier.predict(vec)

        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is True

    def test_convenience_predict_url(self):
        """Test global convenience function predict_url."""
        result = predict_url("https://www.github.com")
        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is True

    def test_get_classifier_status(self):
        """Test operational status retrieval."""
        status = get_classifier_status()
        assert "is_model_available" in status
        assert status["is_model_available"] is True
        assert "metadata" in status


class TestInvalidInputsAndFallbacks:
    """Test suite for edge cases, missing models, and invalid features."""

    def test_missing_model_graceful_handling(self, tmp_path: Path):
        """Test that a classifier with nonexistent paths handles inference safely."""
        missing_model = tmp_path / "nonexistent" / "model.pkl"
        missing_meta = tmp_path / "nonexistent" / "meta.json"

        classifier = MaliciousURLClassifier(model_path=missing_model, metadata_path=missing_meta)
        result = classifier.predict("https://www.example.com")

        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is False
        assert result.predicted_label == "Unavailable"
        assert result.error_message is not None

    def test_invalid_feature_vector_dimension(self, default_classifier: MaliciousURLClassifier):
        """Test that a vector with incorrect length (e.g. 5 instead of 15) is rejected."""
        short_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = default_classifier.predict(short_vec)

        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is False
        assert result.predicted_label == "Error"
        assert "Invalid feature vector length" in (result.error_message or "")

    def test_unsupported_input_type(self, default_classifier: MaliciousURLClassifier):
        """Test that passing integers or unexpected types returns an error result."""
        result = default_classifier.predict(12345)  # type: ignore

        assert isinstance(result, MLPredictionResult)
        assert result.is_model_available is False
        assert result.predicted_label == "Error"


class TestEvaluationMetrics:
    """Test suite for evaluation metric computations and visualizations."""

    def test_compute_metrics_perfect_predictions(self):
        """Test classification metrics computation for known labels."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["confusion_matrix"]["tn"] == 2
        assert metrics["confusion_matrix"]["tp"] == 2
        assert metrics["confusion_matrix"]["fp"] == 0
        assert metrics["confusion_matrix"]["fn"] == 0

    def test_compute_metrics_imperfect_predictions(self):
        """Test metrics computation with false positives and false negatives."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.5
        assert metrics["confusion_matrix"]["tn"] == 1
        assert metrics["confusion_matrix"]["fp"] == 1
        assert metrics["confusion_matrix"]["fn"] == 1
        assert metrics["confusion_matrix"]["tp"] == 1

    def test_plotly_confusion_matrix_generation(self):
        """Test Plotly confusion matrix figure generation."""
        cm = {"matrix": [[10, 2], [1, 15]], "tn": 10, "fp": 2, "fn": 1, "tp": 15}
        fig = plot_confusion_matrix(cm)
        assert fig is not None
        assert len(fig.data) > 0

    def test_plotly_feature_importance_generation(self):
        """Test Plotly feature importance bar chart generation."""
        names = ["feat_a", "feat_b", "feat_c"]
        weights = [0.6, 0.3, 0.1]
        fig = plot_feature_importance(names, weights)
        assert fig is not None
        assert len(fig.data) > 0
