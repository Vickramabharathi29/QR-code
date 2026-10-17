# QRShield Data Directory

This directory stores datasets and sample test files used for URL classification and QR decoding benchmarking.

## Recommended Data Assets:
1. `malicious_urls.csv`: Curated malicious and phishing URL datasets (e.g., from PhishTank, OpenPhish, URLhaus).
2. `benign_urls.csv`: Verified legitimate domains and URLs (e.g., Tranco top 1M, Alexa top 1M).
3. `qr_samples/`: Test image files (PNG, JPG, WebP) with legitimate and malicious QR codes for decoder validation.

## Note:
Raw large CSV files and dataset dumps are excluded from git tracking via `.gitignore`.
