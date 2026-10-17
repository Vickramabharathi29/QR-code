# QRShield: QR Scam and Malicious Redirection Detection

> **Adaptive Threat Prevention Using Reinforcement Learning for Classification and Scam Neutralization**

---

## 📌 Project Overview

QR codes are increasingly exploited in phishing attacks (**quishing**) and malicious redirection schemes. Traditional static scanners only inspect the raw decoded string, leaving users vulnerable to delayed redirects, dynamic QR swaps, and evasion tactics.

**QRShield** is an intelligent detection and prevention system that:
1. **Decodes QR codes** accurately from various visual formats using computer vision (OpenCV & Pyzbar).
2. **Traces full HTTP/HTTPS redirection chains** to uncover hidden destinations and evasive hopping.
3. **Extracts comprehensive lexical and structural features** from both the decoded and resolved URLs.
4. **Classifies malicious intent** via a trained **Random Forest** model with confidence scoring.
5. **Applies an Adaptive Q-Learning Agent** that selects optimal security actions (`ALLOW`, `WARN_USER`, `DEEP_INSPECT`, `BLOCK`) to minimize user disruption while preventing scam execution.
6. **Stores telemetry and feedback in Supabase PostgreSQL** with Row Level Security (RLS) for audit trails and reinforcement learning fine-tuning.

---

## 📂 Project Structure

```text
QR-code/
├── app.py                      # Streamlit web interface entry point
├── config.py                   # Environment configuration, paths & RL parameters
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables template
├── .gitignore                  # Git exclusions (prevents credential & cache leaks)
├── README.md                   # Project documentation
├── supabase/
│   └── schema.sql              # Supabase PostgreSQL DDL, indices & RLS policies
├── modules/
│   ├── __init__.py             # Module package marker
│   ├── qr_decoder.py           # QR decoding using OpenCV & Pyzbar
│   ├── url_analyzer.py         # HTTP redirection tracer & endpoint resolver
│   ├── feature_extractor.py    # URL lexical, structural, and behavioral feature extraction
│   ├── ml_classifier.py        # Random Forest malicious URL classifier
│   ├── reinforcement_learning.py # Q-Learning agent, Q-table & state-action transitions
│   ├── threat_prevention.py    # Threat mitigation engine (Allow / Warn / Inspect / Block)
│   ├── database.py             # Supabase official client manager & query handling
│   └── evaluation.py           # ML and RL performance metrics evaluation
├── data/
│   ├── README.md               # Guide for datasets & QR test samples
│   └── .gitkeep
├── models/
│   ├── README.md               # Guide for model checkpoints & Q-table exports
│   └── .gitkeep
└── tests/
    └── __init__.py             # Pytest test suite
```

---

## 🛠️ Technology Stack

- **Interface:** [Streamlit](https://streamlit.io/)
- **Visual Analytics:** [Plotly](https://plotly.com/)
- **Computer Vision:** [OpenCV](https://opencv.org/) & [Pyzbar](https://pypi.org/project/pyzbar/)
- **Data Science:** [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (Random Forest)
- **Reinforcement Learning:** Tabular Q-Learning (Custom Markov Decision Process)
- **Database:** [Supabase](https://supabase.com/) (Managed PostgreSQL)
- **Testing:** [Pytest](https://pytest.org/)

---

## 🗄️ Supabase Setup & Security

QRShield uses Supabase PostgreSQL for persistence and does not use SQLite.

### 1. Database Schema Execution
1. Navigate to your Supabase project dashboard.
2. Go to the **SQL Editor**.
3. Copy the contents of [`supabase/schema.sql`](supabase/schema.sql) and execute it.
4. This creates:
   - `scan_history`: Raw scan records and timestamps.
   - `threat_analysis_results`: Redirection trails and ML confidence scores.
   - `prevention_decisions`: RL actions taken, Q-values, and state keys.
   - `rl_feedback`: User actions and ground-truth validation for agent tuning.
   - **Row Level Security (RLS)** policies enabled across all tables.

### 2. Environment Variables Configuration
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Fill in your `SUPABASE_URL` and `SUPABASE_KEY` (Anon Public Key):
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-supabase-anon-key
   ```
3. The `.env` file is included in `.gitignore` and must **never** be committed.

---

## 🚀 Installation & Running

```bash
# 1. Create and activate a Python virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

---

## 🔒 Security Best Practices
- **Never commit credentials:** API keys and connection secrets reside solely in the local `.env`.
- **Least Privilege:** Web client interacts only via the Supabase Anon public key regulated by RLS policies.
- **Fail-safe Fallback:** If Supabase is unreachable, the system gracefully notifies the user without crashing or writing to unauthorized fallback databases.
