# QRShield Models Directory

This directory stores trained machine learning models, feature scaler pipelines, and reinforcement learning policy tables.

## Model Artifacts:
1. `random_forest_model.pkl`: Serialized Scikit-learn Random Forest classifier trained on URL lexical, structural, and redirection features.
2. `feature_scaler.pkl`: Feature preprocessor / standardizer pipeline.
3. `q_table.json`: Serialized Q-table storing state-action utility values for adaptive threat prevention decisions.

## Note:
Binary weights (`.pkl`, `.joblib`) are excluded from git tracking via `.gitignore`.
