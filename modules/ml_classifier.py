"""Machine Learning Classifier Module for QRShield.

Uses Scikit-learn's RandomForestClassifier to classify extracted URL features into
Benign or Malicious categories with probability estimates and feature explanations.

Features:
- Strict preservation of the 15 canonical features from modules.feature_extractor.
- Offline training and persistent model serialization (joblib + metadata JSON).
- Fast inference with in-memory cached model singleton.
- Detailed prediction outputs (predicted class, uncalibrated risk score 0-100,
  confidence %, top contributing features, and limitations notice).
- Full integration with modules.evaluation for performance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import config
from modules.evaluation import compute_classification_metrics
from modules.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
    extract_feature_vector,
    extract_url_features,
)

logger = logging.getLogger("qrshield.ml_classifier")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MODEL_PATH = config.MODELS_DIR / "random_forest_model.pkl"
DEFAULT_METADATA_PATH = config.MODELS_DIR / "model_metadata.json"
DEFAULT_DATASET_PATH = config.DATA_DIR / "sample_urls.csv"

# Model limitations statement
MODEL_LIMITATIONS_DISCLAIMER = (
    "Notice on Model Limitations: This Random Forest classifier performs static lexical "
    "and structural analysis. It cannot detect zero-day domains, compromised legitimate servers, "
    "or runtime server-side cloaking without dynamic inspection. The current model was trained "
    "on a curated development dataset and should be augmented with enterprise feeds for production."
)


@dataclass
class MLPredictionResult:
    """Detailed outcome of a Random Forest classification inference."""

    predicted_label: str  # 'Benign' or 'Malicious'
    predicted_class: int  # 0 or 1
    risk_score: float  # 0.0 to 100.0 (model malicious probability * 100)
    confidence: float  # Confidence percentage (e.g. 92.5%)
    probabilities: Dict[str, float]  # {'Benign': 0.08, 'Malicious': 0.92}
    top_contributing_features: List[Dict[str, Any]] = field(default_factory=list)
    limitations_notice: str = MODEL_LIMITATIONS_DISCLAIMER
    model_version: str = "v1.0.0"
    is_model_available: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary format."""
        return {
            "predicted_label": self.predicted_label,
            "predicted_class": self.predicted_class,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "top_contributing_features": self.top_contributing_features,
            "limitations_notice": self.limitations_notice,
            "model_version": self.model_version,
            "is_model_available": self.is_model_available,
        }


class MaliciousURLClassifier:
    """Random Forest classifier for detecting malicious, scam, and phishing URLs."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
        self.feature_extractor = FeatureExtractor()
        self.model: Optional[RandomForestClassifier] = None
        self.metadata: Dict[str, Any] = {}

    def is_trained_model_available(self) -> bool:
        """Check if serialized model and metadata exist on disk."""
        return self.model_path.exists() and self.metadata_path.exists()

    def train(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """Train Random Forest classifier on a labelled URL dataset and save to disk.

        Args:
            dataset_path: Path to CSV dataset with 'url' and 'label' columns.
            test_size: Ratio of held-out test data for evaluation.
            random_state: Seed for reproducible split and tree ensemble.
            n_estimators: Number of decision trees.
            max_depth: Maximum tree depth.

        Returns:
            Dict[str, Any]: Training summary including evaluation metrics and metadata.
        """
        data_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        if not data_path.exists():
            raise FileNotFoundError(f"Training dataset not found at: {data_path}")

        logger.info(f"Loading training dataset from: {data_path}")
        df = pd.read_csv(data_path)

        if "url" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset CSV must contain 'url' and 'label' columns.")

        # Clean rows
        df = df.dropna(subset=["url", "label"]).copy()
        df["label"] = df["label"].astype(int)

        logger.info(f"Extracting features for {len(df)} URLs...")
        # Extract features for all URLs
        X_vectors: List[List[float]] = []
        for u in df["url"]:
            X_vectors.append(self.feature_extractor.to_vector(self.feature_extractor.extract_features(str(u))))

        X = np.array(X_vectors, dtype=float)
        y = df["label"].values

        # Stratified train/test split to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(f"Training RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Evaluate on held-out test split
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        eval_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        # Feature importances
        importances = [float(imp) for imp in clf.feature_importances_]
        feature_importance_dict = {
            name: round(imp, 4) for name, imp in zip(FEATURE_NAMES, importances)
        }

        # Assemble metadata
        metadata = {
            "model_type": "RandomForestClassifier",
            "model_version": "v1.0.0",
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "feature_names": FEATURE_NAMES,
            "feature_importances": feature_importance_dict,
            "dataset_info": {
                "source": str(data_path.name),
                "total_samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "class_balance": {
                    "benign": int(np.sum(y == 0)),
                    "malicious": int(np.sum(y == 1)),
                },
                "disclaimer": (
                    "Trained on curated sample data for development testing. "
                    "Production deployment requires enterprise threat feeds."
                ),
            },
            "evaluation_metrics": eval_metrics,
        }

        # Save artifacts to disk
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, self.model_path)
        logger.info(f"Saved trained model to: {self.model_path}")

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to: {self.metadata_path}")

        self.model = clf
        self.metadata = metadata

        return metadata

    def load_model(self, force_reload: bool = False, auto_train: bool = False) -> bool:
        """Load trained Random Forest model and metadata from disk.

        Args:
            force_reload: If True, reloads model even if already in memory.
            auto_train: If True and model is missing, attempts to train from default dataset.

        Returns:
            bool: True if model is loaded and ready for predictions.
        """
        if self.model is not None and not force_reload:
            return True

        if not self.is_trained_model_available():
            if auto_train and DEFAULT_DATASET_PATH.exists():
                logger.info("Model artifacts not found on disk. Auto-training initial model...")
                try:
                    self.train(DEFAULT_DATASET_PATH)
                    return True
                except Exception as exc:
                    logger.error(f"Failed to auto-train initial model: {exc}")
                    return False
            else:
                logger.warning(f"No trained model found at {self.model_path}.")
                return False

        try:
            self.model = joblib.load(self.model_path)
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            logger.info("Successfully loaded Random Forest model and metadata.")
            return True
        except Exception as exc:
            logger.error(f"Error loading model from {self.model_path}: {exc}")
            self.model = None
            return False

    def predict(
        self,
        url_or_features: Union[str, Dict[str, Any], List[float]],
    ) -> MLPredictionResult:
        """Predict whether a URL is Benign or Malicious with confidence score.

        Args:
            url_or_features: URL string, dictionary of features, or 15-dim feature vector.

        Returns:
            MLPredictionResult: Detailed prediction, confidence, risk score, and top indicators.
        """
        if self.model is None and not self.load_model():
            return MLPredictionResult(
                predicted_label="Unavailable",
                predicted_class=0,
                risk_score=0.0,
                confidence=0.0,
                probabilities={"Benign": 0.5, "Malicious": 0.5},
                is_model_available=False,
                error_message="Random Forest model is not trained or loaded. Please train the model.",
            )

        # 1. Prepare numerical feature vector
        raw_feature_dict: Dict[str, Any] = {}
        if isinstance(url_or_features, str):
            raw_feature_dict = self.feature_extractor.extract_features(url_or_features)
            feature_vector = self.feature_extractor.to_vector(raw_feature_dict)
        elif isinstance(url_or_features, dict):
            raw_feature_dict = url_or_features
            feature_vector = self.feature_extractor.to_vector(raw_feature_dict)
        elif isinstance(url_or_features, (list, np.ndarray)):
            feature_vector = [float(v) for v in url_or_features]
            if len(feature_vector) != len(FEATURE_NAMES):
                return MLPredictionResult(
                    predicted_label="Error",
                    predicted_class=0,
                    risk_score=0.0,
                    confidence=0.0,
                    probabilities={"Benign": 0.5, "Malicious": 0.5},
                    is_model_available=False,
                    error_message=f"Invalid feature vector length ({len(feature_vector)}). Expected {len(FEATURE_NAMES)}.",
                )
            raw_feature_dict = {name: val for name, val in zip(FEATURE_NAMES, feature_vector)}
        else:
            return MLPredictionResult(
                predicted_label="Error",
                predicted_class=0,
                risk_score=0.0,
                confidence=0.0,
                probabilities={"Benign": 0.5, "Malicious": 0.5},
                is_model_available=False,
                error_message="Unsupported input format for prediction.",
            )

        # 2. Run model inference
        X_sample = np.array([feature_vector], dtype=float)
        pred_class = int(self.model.predict(X_sample)[0])
        pred_probs = self.model.predict_proba(X_sample)[0]

        prob_benign = float(pred_probs[0])
        prob_malicious = float(pred_probs[1]) if len(pred_probs) > 1 else 0.0

        predicted_label = "Malicious" if pred_class == 1 else "Benign"
        risk_score = round(prob_malicious * 100.0, 1)
        confidence = round(max(prob_benign, prob_malicious) * 100.0, 1)

        # 3. Identify top contributing features for this sample
        importances = self.metadata.get("feature_importances", {})
        top_features: List[Dict[str, Any]] = []

        for name, val in raw_feature_dict.items():
            imp = importances.get(name, 0.0)
            # Give higher priority to features with active flags and high global importance
            active_weight = imp * (1.5 if (val and val != 0) else 0.2)
            top_features.append({
                "feature": name,
                "value": val,
                "importance": imp,
                "influence_rank": active_weight,
            })

        top_features.sort(key=lambda x: x["influence_rank"], reverse=True)
        selected_top = top_features[:5]

        return MLPredictionResult(
            predicted_label=predicted_label,
            predicted_class=pred_class,
            risk_score=risk_score,
            confidence=confidence,
            probabilities={"Benign": round(prob_benign, 4), "Malicious": round(prob_malicious, 4)},
            top_contributing_features=selected_top,
            model_version=self.metadata.get("model_version", "v1.0.0"),
            is_model_available=True,
            error_message=None,
        )


# Global singleton instance
_classifier_instance = MaliciousURLClassifier()


def predict_url(url: str) -> MLPredictionResult:
    """Convenience helper to classify a URL using the default classifier instance."""
    return _classifier_instance.predict(url)


def get_classifier_status() -> Dict[str, Any]:
    """Retrieve operational status, model availability, and metadata."""
    loaded = _classifier_instance.load_model()
    return {
        "is_model_available": loaded,
        "model_path": str(_classifier_instance.model_path),
        "metadata": _classifier_instance.metadata,
    }


def train_classifier(dataset_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Train or retrain the Random Forest model."""
    return _classifier_instance.train(dataset_path=dataset_path)
