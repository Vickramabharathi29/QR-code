# QRShield Data Directory

This directory stores datasets and sample test files used for URL classification and QR decoding benchmarking.

## Datasets:
1. `sample_urls.csv`: A development dataset comprising 120 curated URLs (60 benign, 60 malicious) with label `0` (benign) and `1` (malicious).
   - **Important Disclaimer**: This sample dataset is provided strictly for local development, pipeline validation, and unit testing. It is **not** a substitute for real-world enterprise threat intelligence feeds (such as PhishTank, OpenPhish, URLhaus, or Tranco top 1M) and should not be used to claim real-world production detection accuracy.
2. `qr_samples/`: Test image files (PNG) with legitimate and malicious QR codes for decoder validation.

## Data Pipeline:
- The data loader in `modules/ml_classifier.py` reads CSV datasets with `url` and `label` columns.
- Features are extracted statically using `modules/feature_extractor.py`.
- Training and evaluation splits are strictly separated (e.g. 75% train, 25% test stratified) to prevent data leakage.
- Raw external CSV dumps should be placed here and are excluded from git tracking via `.gitignore`.
