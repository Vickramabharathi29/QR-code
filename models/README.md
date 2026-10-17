# QRShield Models Directory

This directory stores trained machine learning models, feature scaler pipelines, and reinforcement learning policy tables.

## Model Artifacts:
1. `random_forest_model.pkl`: Serialized Scikit-learn Random Forest classifier trained on URL lexical and structural features.
2. `model_metadata.json`: Feature schema, feature importances, dataset summary, and held-out test evaluation metrics.
3. `feature_scaler.pkl`: Feature preprocessor / standardizer pipeline (optional).
4. `q_table.json`: Serialized Q-table storing state-action utility values for adaptive threat prevention decisions (Step 6).

## Note:
Binary weights (`.pkl`, `.joblib`) are excluded from git tracking via `.gitignore`.
