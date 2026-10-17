"""QRShield: QR Scam and Malicious Redirection Detection.

Adaptive Threat Prevention Using Reinforcement Learning & Machine Learning.
Streamlit Web Application Entry Point.
"""

import io
import uuid
from urllib.parse import urlparse
from PIL import Image
import pandas as pd
import streamlit as st

import importlib
import config
from modules import database
import modules.evaluation
import modules.reinforcement_learning
import modules.threat_prevention
importlib.reload(database)
importlib.reload(modules.evaluation)
importlib.reload(modules.reinforcement_learning)
importlib.reload(modules.threat_prevention)

from modules.evaluation import (
    plot_action_distribution,
    plot_cumulative_rewards,
    plot_confusion_matrix,
    plot_feature_importance,
)
from modules.ml_classifier import (
    MODEL_LIMITATIONS_DISCLAIMER,
    MLPredictionResult,
    get_classifier_status,
    predict_url,
)
from modules.qr_decoder import (
    QRScanResult,
    decode_qr,
    get_decoder_status,
)
from modules.reinforcement_learning import (
    ACTIONS as RL_ACTIONS,
    QLearningAgent,
    get_rl_agent,
)
from modules.threat_prevention import (
    PreventionDecision,
    ThreatPreventionEngine,
    get_prevention_engine,
)
from modules.url_analyzer import URLAnalysisResult, analyze_url

# Upload size configuration with safe default
MAX_UPLOAD_SIZE_MB = getattr(config, "MAX_UPLOAD_SIZE_MB", 5.0)

# Page configuration
st.set_page_config(
    page_title="QRShield | Threat Scanner & URL Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "scan_result" not in st.session_state:
    st.session_state["scan_result"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "last_uploaded_filename" not in st.session_state:
    st.session_state["last_uploaded_filename"] = None
if "saved_scan_tokens" not in st.session_state:
    st.session_state["saved_scan_tokens"] = set()
if "last_saved_scan_id" not in st.session_state:
    st.session_state["last_saved_scan_id"] = None
if "submitted_feedback_scans" not in st.session_state:
    st.session_state["submitted_feedback_scans"] = set()

# Custom CSS for modern cybersecurity styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .scanner-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-https {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-http {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-danger {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-special {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-text {
        background-color: #334155;
        color: #e2e8f0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-risk-low {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-risk-med {
        background-color: #78350f;
        color: #fcd34d;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #d97706;
    }
    .badge-risk-high {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #dc2626;
    }
    .risk-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .badge-ml-benign {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-ml-malicious {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #dc2626;
    }
    .assessment-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .badge-action-allow {
        background-color: #064e3b;
        color: #34d399;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-action-warn {
        background-color: #78350f;
        color: #fbbf24;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #d97706;
    }
    .badge-action-block {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #dc2626;
    }
    .prevention-card {
        background-color: #0b1329;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .model-limitations-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_scan():
    """Clear scanned results and reset file uploader."""
    st.session_state["scan_result"] = None
    st.session_state["uploader_key"] += 1
    st.session_state["last_uploaded_filename"] = None
    st.rerun()


# =============================================================================
# Sidebar: System Diagnostics & Settings
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Diagnostics")

    # 1. Supabase Status
    if not database.is_database_configured():
        st.warning("🟡 Supabase Not Configured (`.env`)")
    else:
        connected, conn_msg = database.test_connection()
        if connected:
            st.success("🟢 Supabase Connected & Ready")
        elif "schema.sql" in conn_msg:
            st.warning("🟡 Supabase Connected (Tables Pending)")
            st.caption("Execute `supabase/schema.sql` in Supabase SQL Editor.")
        else:
            st.error(f"🔴 Supabase Error: {conn_msg[:50]}...")

    st.divider()

    # 2. QR Decoder Engine Status
    st.markdown("### 📷 Decoder Engines")
    dec_status = get_decoder_status()
    st.markdown(f"**Primary Engine:** `{dec_status['primary_engine']}`")
    st.markdown("- **OpenCV Detector:** 🟢 Active")

    if dec_status["pyzbar_available"]:
        st.markdown("- **Pyzbar Engine:** 🟢 Active")
    else:
        st.markdown("- **Pyzbar Engine:** 🟡 Inactive (OpenCV Fallback Active)")
        with st.expander("ℹ️ Pyzbar Windows Note"):
            st.caption(
                "Pyzbar uses `libzbar-64.dll`, which requires the "
                "**Microsoft Visual C++ 2013 Redistributable (x64)**. "
                "OpenCV's native QRCodeDetector is handling all QR detection automatically."
            )

    st.divider()

    # 3. Machine Learning Engine Status
    st.markdown("### 🌲 ML Classifier Engine")
    ml_status = get_classifier_status()
    if ml_status["is_model_available"]:
        st.markdown("- **Model:** `RandomForestClassifier`")
        st.markdown("- **Status:** 🟢 `Trained & Ready`")
        meta = ml_status.get("metadata", {})
        eval_metrics = meta.get("evaluation_metrics", {})
        test_acc = eval_metrics.get("accuracy")
        if test_acc is not None:
            st.markdown(f"- **Test Accuracy:** `{test_acc * 100:.1f}%`")
        d_info = meta.get("dataset_info", {})
        st.markdown(f"- **Sample Dataset:** `{d_info.get('total_samples', 120)} URLs (Curated)`")
    else:
        st.markdown("- **Status:** 🟡 `Model Not Loaded`")

    st.divider()

    # 4. Reinforcement Learning Engine Status
    st.markdown("### 🧠 Adaptive RL Engine")
    rl_agent = get_rl_agent()
    if rl_agent.is_policy_loaded():
        st.markdown("- **Algorithm:** `Tabular Q-Learning`")
        st.markdown("- **Status:** 🟢 `Policy Active & Loaded`")
        st.markdown(f"- **States in Q-Table:** `{len(rl_agent.q_table)} Discrete States`")
        total_ep = rl_agent.training_metadata.get("episodes", 400)
        cum_r = rl_agent.training_metadata.get("results", {}).get("total_cumulative_reward", 0.0)
        st.markdown(f"- **Trained Episodes:** `{total_ep}` | **Reward:** `+{cum_r:.1f}`")
    else:
        st.markdown("- **Status:** 🟡 `Q-Table Not Loaded`")

    with st.expander("⚡ Run Controlled Q-Learning Training", expanded=False):
        st.caption("Trains policy on labelled ground truth scenarios. Random Forest classifier is preserved without retraining.")
        ep_slider = st.slider("Simulation Episodes", min_value=100, max_value=1000, value=400, step=50, key="rl_ep_slider")
        lr_slider = st.slider("Learning Rate (α)", min_value=0.05, max_value=0.5, value=0.1, step=0.05, key="rl_lr_slider")
        
        if st.button("🚀 Train / Retrain Q-Learning Agent", use_container_width=True, type="primary"):
            with st.spinner(f"Training Q-Learning policy across {ep_slider} episodes..."):
                rl_agent.lr = lr_slider
                train_meta = rl_agent.train_simulation(episodes=ep_slider)
                st.success(f"✅ Training completed! Cumulative Reward: {train_meta['results']['total_cumulative_reward']:.1f}")
                st.rerun()

    st.divider()

    # 5. Pipeline Roadmap
    st.markdown("### 🗺️ Pipeline Progress")
    st.markdown("1. ✅ **Project Architecture & Setup**")
    st.markdown("2. ✅ **Supabase Database Configuration**")
    st.markdown("3. ✅ **QR Image Scanner**")
    st.markdown("4. ✅ **URL Feature Extractor & Suspicious Analyzer**")
    st.markdown("5. ✅ **Random Forest Malicious Classifier**")
    st.markdown("6. ✅ **Adaptive Threat Prevention Engine**")
    st.markdown("7. 🟢 **Persistent Supabase Integration (Active)**")

    st.divider()
    if st.button("🔄 Reset / New Scan", use_container_width=True):
        reset_scan()


# =============================================================================
# Main Header
# =============================================================================
st.markdown('<div class="main-header">🛡️ QRShield</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-caption">'
    "QR Scam & Malicious Redirection Detection with Adaptive Threat Prevention"
    "</div>",
    unsafe_allow_html=True,
)

# Safety Protocol Banner
st.caption(
    "🔒 **Safety Protocol Active:** QRShield decodes and statically inspects QR payloads "
    "in an isolated sandbox. It never executes code, downloads remote assets, or follows redirects."
)

st.markdown("---")

# =============================================================================
# Navigation Tabs: Real-Time Scanner vs. Supabase Scan History
# =============================================================================
tab_scanner, tab_history = st.tabs([
    "🔍 Real-Time QR Scanner",
    "📜 Supabase Scan History & Security Audit Log",
])

with tab_scanner:
    # =============================================================================
    # Two-Column Layout: Scanner Controls (Left) & Results (Right)
    # =============================================================================
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📤 Upload QR Code")
        st.markdown(
            f"Select a QR code image to inspect. Supported formats: **PNG, JPG, JPEG** "
            f"(Max file size: **{MAX_UPLOAD_SIZE_MB:.0f} MB**)."
        )

        uploaded_file = st.file_uploader(
            label="Drop QR image here or click to browse",
            type=["png", "jpg", "jpeg"],
            key=f"qr_uploader_{st.session_state['uploader_key']}",
            help="Upload clear QR code images. High-contrast and well-lit images yield best results.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            file_size_kb = len(file_bytes) / 1024

            # Validate file size against configured limit
            if file_size_mb > MAX_UPLOAD_SIZE_MB:
                st.error(
                    f"❌ **File Size Limit Exceeded:** `{uploaded_file.name}` is `{file_size_mb:.2f} MB`, "
                    f"which exceeds the maximum allowed limit of `{MAX_UPLOAD_SIZE_MB:.1f} MB`."
                )
                st.info(f"💡 Please select an image under {MAX_UPLOAD_SIZE_MB:.0f} MB or compress the file.")
                if st.button("🔄 Clear", use_container_width=True):
                    reset_scan()
            else:
                # Image Metadata & Preview
                st.markdown(f"**Filename:** `{uploaded_file.name}` | **Size:** `{file_size_kb:.1f} KB`")

                try:
                    image_preview = Image.open(io.BytesIO(file_bytes))
                    st.image(
                        image_preview,
                        caption=f"Preview: {uploaded_file.name} ({image_preview.width}x{image_preview.height} px)",
                        use_container_width=True,
                    )
                except Exception:
                    st.error("❌ Failed to display preview. The uploaded image file may be corrupted.")

                # Decode Action Button
                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    decode_clicked = st.button("🔍 Decode & Analyze QR", type="primary", use_container_width=True)
                with btn_col2:
                    if st.button("🔄 Clear", use_container_width=True):
                        reset_scan()

                if decode_clicked:
                    with st.spinner("Decoding QR code and extracting features..."):
                        scan_result = decode_qr(file_bytes, filename=uploaded_file.name)
                        st.session_state["scan_result"] = scan_result
                        st.session_state["last_uploaded_filename"] = uploaded_file.name

        else:
            st.info("👈 Upload an image on the left to begin QR analysis.")


    with col_right:
        st.markdown("### 📋 Threat Analysis & Feature Inspection")

        scan_result: QRScanResult = st.session_state["scan_result"]

        if scan_result is None:
            st.markdown(
                """
                <div style="border: 2px dashed #334155; border-radius: 12px; padding: 3rem 1.5rem; text-align: center; color: #64748b;">
                    <h4>No Scan Performed Yet</h4>
                    <p>Upload a QR code image and click <strong>Decode & Analyze QR</strong> to inspect its payload safely.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif not scan_result.success:
            st.error(f"❌ **Validation Error:** {scan_result.error_message}")

        elif scan_result.qr_count == 0:
            st.warning(f"⚠️ {scan_result.error_message}")
            st.info(
                "💡 **Tip:** Ensure the entire QR pattern is visible, properly focused, and not obscured by harsh shadows or reflections."
            )

        else:
            st.success(f"✅ Successfully decoded **{scan_result.qr_count}** QR code(s) via **{scan_result.engine_used}**")

            for idx, item in enumerate(scan_result.items):
                if scan_result.qr_count > 1:
                    st.markdown(f"#### 🏷️ QR Code #{idx + 1}")

                # Run static URL & suspicious feature analysis
                analysis: URLAnalysisResult = analyze_url(item.raw_payload)

                # 1. Payload Classification Badges
                badge_html = ""
                if item.payload_type == "URL_HTTPS":
                    badge_html = '<span class="badge-https">🔒 HTTPS URL</span>'
                elif item.payload_type == "URL_HTTP":
                    badge_html = '<span class="badge-http">⚠️ HTTP URL (Insecure Cleartext)</span>'
                elif item.payload_type == "DANGEROUS_SCHEME":
                    badge_html = f'<span class="badge-danger">🚨 DANGEROUS URI ({item.scheme.upper() if item.scheme else "UNKNOWN"})</span>'
                elif item.payload_type == "SPECIALIZED_SCHEME":
                    badge_html = f'<span class="badge-special">📱 SPECIALIZED URI ({item.scheme.upper() if item.scheme else "CUSTOM"})</span>'
                else:
                    badge_html = '<span class="badge-text">📄 Plain Text</span>'

                status_badge = f"<code>{analysis.status}</code>"
                st.markdown(f"**Payload Type:** {badge_html} &nbsp;|&nbsp; **Validation Status:** {status_badge}", unsafe_allow_html=True)
                st.caption(f"Status Note: {analysis.validation_message}")

                # 2. Decoded Raw Payload Display
                st.markdown("**Decoded Content (Safe Static View):**")
                st.code(item.raw_payload, language="text")

                # 3. Dual Threat Assessment: Heuristics & Machine Learning
                st.markdown("#### ⚖️ Dual Threat Assessment: Heuristics & Machine Learning")

                is_valid_url = (item.payload_type in ["URL_HTTPS", "URL_HTTP"]) and (analysis.status != "MALFORMED")

                dual_col1, dual_col2 = st.columns(2)

                with dual_col1:
                    st.markdown(
                        """
                        <div class="assessment-card">
                            <h5 style="margin-top:0; color:#38bdf8;">⚡ Rule-Based Heuristic Risk</h5>
                            <p style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.5rem;">
                                Deterministically inspects known risk patterns, raw IP hosts, keywords, and format anomalies.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    risk_badge = ""
                    if analysis.risk_category == "HIGH":
                        risk_badge = f'<span class="badge-risk-high">🔴 HIGH RISK ({analysis.risk_score:.2f})</span>'
                    elif analysis.risk_category == "MEDIUM":
                        risk_badge = f'<span class="badge-risk-med">🟡 MEDIUM RISK ({analysis.risk_score:.2f})</span>'
                    else:
                        risk_badge = f'<span class="badge-risk-low">🟢 LOW RISK ({analysis.risk_score:.2f})</span>'

                    st.markdown(f"**Heuristic Rating:** {risk_badge}", unsafe_allow_html=True)
                    st.progress(analysis.risk_score)

                    if analysis.risk_indicators:
                        st.markdown("**Triggered Indicators:**")
                        for ind in analysis.risk_indicators:
                            st.markdown(f"- <span style='font-size:0.86rem;'>{ind}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("No suspicious heuristic rules triggered.")

                with dual_col2:
                    st.markdown(
                        """
                        <div class="assessment-card">
                            <h5 style="margin-top:0; color:#38bdf8;">🌲 Random Forest ML Classifier (Step 5)</h5>
                            <p style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.5rem;">
                                Ensemble statistical classification trained on 15 extracted lexical and structural features.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if not is_valid_url:
                        st.warning(
                            "⚠️ **Classification Skipped:** Machine learning URL classification requires "
                            "a syntactically valid HTTP or HTTPS URL. Plain text, custom schemes, or malformed URIs "
                            "cannot be evaluated by the model."
                        )
                    else:
                        ml_pred: MLPredictionResult = predict_url(item.raw_payload)

                        if not ml_pred.is_model_available:
                            st.error(f"❌ {ml_pred.error_message}")
                        else:
                            if ml_pred.predicted_label == "Malicious":
                                ml_badge = f'<span class="badge-ml-malicious">🚨 MALICIOUS (Risk: {ml_pred.risk_score:.1f}/100)</span>'
                            else:
                                ml_badge = f'<span class="badge-ml-benign">🛡️ BENIGN (Risk: {ml_pred.risk_score:.1f}/100)</span>'

                            st.markdown(f"**Predicted Class:** {ml_badge}", unsafe_allow_html=True)
                            st.progress(float(ml_pred.risk_score) / 100.0)

                            ml_sub1, ml_sub2 = st.columns(2)
                            with ml_sub1:
                                st.metric(
                                    "Model Risk Score",
                                    f"{ml_pred.risk_score:.1f} / 100",
                                    help="Derived from Random Forest tree voting fraction (uncalibrated probability * 100).",
                                )
                            with ml_sub2:
                                st.metric(
                                    "Prediction Confidence",
                                    f"{ml_pred.confidence:.1f}%",
                                    help="Ensemble vote percentage for the winning class.",
                                )

                            prob_b = ml_pred.probabilities.get("Benign", 0.0) * 100
                            prob_m = ml_pred.probabilities.get("Malicious", 0.0) * 100
                            st.caption(f"**Tree Votes:** Benign: `{prob_b:.1f}%` &nbsp;|&nbsp; Malicious: `{prob_m:.1f}%`")

                            if ml_pred.top_contributing_features:
                                with st.expander("🔬 Top Feature Indicators for this URL", expanded=False):
                                    df_top = pd.DataFrame([
                                        {
                                            "Feature": f["feature"],
                                            "Value in Sample": f["value"],
                                            "Importance Weight": f"{f['importance']:.4f}",
                                        }
                                        for f in ml_pred.top_contributing_features
                                    ])
                                    st.dataframe(df_top, use_container_width=True, hide_index=True)

                # Model Limitations Notice
                if is_valid_url:
                    st.markdown(
                        f"""
                        <div class="model-limitations-box">
                            <strong>⚠️ Notice on Model Limitations:</strong> {MODEL_LIMITATIONS_DISCLAIMER}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 4. Adaptive Threat Prevention Decision (Q-Learning + Safety Enforcer)
                st.markdown("#### 🛡️ Adaptive Threat Prevention Decision (Q-Learning + Safety Enforcer)")
                prevention_engine = get_prevention_engine()
                decision: PreventionDecision = prevention_engine.evaluate_threat(
                    raw_payload=item.raw_payload,
                    payload_type=item.payload_type,
                    analysis_result=analysis,
                    ml_prediction=ml_pred if is_valid_url else None,
                )

                # Prevention Action Display Card
                action_badge_html = ""
                if decision.final_action == "ALLOW":
                    action_badge_html = '<span class="badge-action-allow">🟢 ALLOW (Proceed Permitted)</span>'
                elif decision.final_action == "WARN":
                    action_badge_html = '<span class="badge-action-warn">🟡 WARN (User Confirmation Required)</span>'
                else:
                    action_badge_html = '<span class="badge-action-block">🔴 BLOCK (Navigation Strictly Prevented)</span>'

                raw_badge = f"<code>{decision.raw_action}</code>"
                st.markdown(
                    f"**Enforced Prevention Action:** {action_badge_html} &nbsp;|&nbsp; "
                    f"**RL Raw Action:** {raw_badge}",
                    unsafe_allow_html=True,
                )

                if decision.is_safety_override:
                    st.warning(f"🛡️ **Safety Policy Intervention:** {decision.override_reason}")

                st.info(f"💡 **Decision Rationale:** {decision.explanation}")

                # State & Q-Values breakdown
                st.markdown(f"**Quantized State Key:** <code>{decision.state_key}</code>", unsafe_allow_html=True)
                q_col1, q_col2, q_col3 = st.columns(3)
                with q_col1:
                    st.metric("Q-Value: ALLOW", f"{decision.q_values.get('ALLOW', 0.0):.2f}")
                with q_col2:
                    st.metric("Q-Value: WARN", f"{decision.q_values.get('WARN', 0.0):.2f}")
                with q_col3:
                    st.metric("Q-Value: BLOCK", f"{decision.q_values.get('BLOCK', 0.0):.2f}")

                # 5. Persistent Supabase Storage & Telemetry (Deliberate Save Option)
                st.markdown("#### 💾 Persistent Supabase Storage")

                db_ready = database.is_database_configured()
                if not db_ready:
                    st.caption(
                        "🟡 **Supabase Not Configured:** Scan results cannot be saved to the cloud database. "
                        "Configure `SUPABASE_URL` and `SUPABASE_KEY` in `.env` to enable persistent scan history."
                    )
                else:
                    scan_token = f"{item.raw_payload}_{st.session_state.get('last_uploaded_filename', 'upload')}_{idx}"
                    is_saved = scan_token in st.session_state.get("saved_scan_tokens", set())

                    s_col1, s_col2 = st.columns([2, 1])
                    with s_col1:
                        save_btn_clicked = st.button(
                            "💾 Save Scan Record to Supabase",
                            key=f"btn_save_scan_{idx}",
                            type="primary" if not is_saved else "secondary",
                            disabled=is_saved,
                            help="Persist scan record, threat analysis, and prevention decision to Supabase."
                        )
                    with s_col2:
                        if is_saved:
                            saved_id = st.session_state.get(f"scan_id_{scan_token}", "Saved")
                            st.caption(f"✅ Record Persisted (`{str(saved_id)[:8]}...`)")

                    if save_btn_clicked:
                        with st.spinner("Writing audit record to Supabase PostgreSQL..."):
                            client_scan_uuid = str(uuid.uuid4())

                            # Normalize inputs for schema
                            if is_valid_url and ml_pred and getattr(ml_pred, "is_model_available", True):
                                p_label = getattr(ml_pred, "predicted_label", "SAFE").upper().strip()
                                pred_class = "SAFE" if p_label in ("BENIGN", "SAFE") else ("MALICIOUS" if p_label in ("MALICIOUS", "PHISHING") else "SUSPICIOUS")
                                raw_score = float(getattr(ml_pred, "risk_score", 0.0))
                            else:
                                pred_class = "SAFE" if getattr(item, "payload_type", "OTHER") == "TEXT" else "SUSPICIOUS"
                                raw_score = float(getattr(analysis, "risk_score", 0.0)) if analysis else 0.0

                            normalized_score = raw_score / 100.0 if raw_score > 1.0 else raw_score
                            normalized_score = max(0.0, min(1.0, float(normalized_score)))

                            # Retrieve indicators safely using official risk_indicators with fallbacks
                            indicators = []
                            if analysis is not None:
                                raw_ind = getattr(analysis, "risk_indicators", getattr(analysis, "suspicious_indicators", []))
                                if isinstance(raw_ind, list):
                                    indicators = [str(x) for x in raw_ind if x]
                                elif raw_ind:
                                    indicators = [str(raw_ind)]

                            save_result = database.save_complete_scan(
                                extracted_url=getattr(item, "raw_payload", None),
                                qr_payload_type=getattr(item, "payload_type", "OTHER"),
                                predicted_class=pred_class,
                                risk_score=normalized_score,
                                detected_indicators=indicators,
                                selected_action=getattr(decision, "final_action", "WARN"),
                                decision_reason=getattr(decision, "explanation", "Automated threat evaluation"),
                                scan_id=client_scan_uuid,
                                model_version=ml_status.get("metadata", {}).get("model_version", "v1.0.0") if ml_status else "v1.0.0",
                                policy_version=getattr(decision, "policy_version", "v1.0.0"),
                            )

                            if save_result:
                                st.session_state.setdefault("saved_scan_tokens", set()).add(scan_token)
                                st.session_state[f"scan_id_{scan_token}"] = save_result["scan_id"]
                                st.session_state["last_saved_scan_id"] = save_result["scan_id"]
                                st.success(
                                    f"✅ **Record Saved to Supabase!** Scan ID: `{save_result['scan_id'][:8]}...` "
                                    f"— Audit log updated in `scan_history`, `threat_analysis`, and `prevention_decisions`."
                                )
                            else:
                                st.error(
                                    "❌ **Database Write Failed:** Could not persist scan record. "
                                    "Verify table schema and network connectivity in the System Diagnostics sidebar."
                                )

                    # Reinforcement Learning Feedback Expander
                    with st.expander("💬 Submit Controlled Evaluation Feedback (RL Feedback)", expanded=False):
                        st.caption(
                            "Submit human-in-the-loop feedback on the prevention action for controlled evaluation. "
                            "Feedback is stored strictly as unverified telemetry (`source='USER'`) in `rl_feedback` "
                            "and does not automatically retrain the model or alter verified ground-truth."
                        )

                        fb_action_options = ["ALLOW", "WARN", "BLOCK"]
                        default_idx = 0 if decision.final_action == "ALLOW" else (1 if decision.final_action == "WARN" else 2)
                        fb_sel_action = st.selectbox(
                            "Selected Action Evaluation:",
                            fb_action_options,
                            index=default_idx,
                            key=f"fb_act_{idx}",
                            help="What action should have been taken for this payload?"
                        )

                        fb_gt_options = ["BENIGN", "MALICIOUS", "SUSPICIOUS", "UNKNOWN"]
                        fb_gt = st.selectbox(
                            "Observed Ground-Truth Threat Category:",
                            fb_gt_options,
                            index=0 if decision.final_action == "ALLOW" else 1,
                            key=f"fb_gt_{idx}",
                        )

                        fb_reward_val = st.slider(
                            "Assigned Utility / Reward Score:",
                            min_value=-10.0,
                            max_value=10.0,
                            value=2.0 if fb_sel_action == decision.final_action else -5.0,
                            step=0.5,
                            key=f"fb_reward_{idx}",
                            help="Positive reward reinforces the action for this state; negative reward penalizes it."
                        )

                        fb_btn = st.button("📤 Submit RL Feedback to Supabase", key=f"fb_submit_{idx}")
                        if fb_btn:
                            active_scan_id = st.session_state.get(f"scan_id_{scan_token}")
                            if not active_scan_id:
                                client_scan_uuid = str(uuid.uuid4())
                                if is_valid_url and ml_pred and getattr(ml_pred, "is_model_available", True):
                                    p_label = getattr(ml_pred, "predicted_label", "SAFE").upper().strip()
                                    pred_class = "SAFE" if p_label in ("BENIGN", "SAFE") else ("MALICIOUS" if p_label in ("MALICIOUS", "PHISHING") else "SUSPICIOUS")
                                    raw_score = float(getattr(ml_pred, "risk_score", 0.0))
                                else:
                                    pred_class = "SAFE" if getattr(item, "payload_type", "OTHER") == "TEXT" else "SUSPICIOUS"
                                    raw_score = float(getattr(analysis, "risk_score", 0.0)) if analysis else 0.0

                                normalized_score = raw_score / 100.0 if raw_score > 1.0 else raw_score
                                normalized_score = max(0.0, min(1.0, float(normalized_score)))

                                # Retrieve indicators safely using official risk_indicators with fallbacks
                                indicators = []
                                if analysis is not None:
                                    raw_ind = getattr(analysis, "risk_indicators", getattr(analysis, "suspicious_indicators", []))
                                    if isinstance(raw_ind, list):
                                        indicators = [str(x) for x in raw_ind if x]
                                    elif raw_ind:
                                        indicators = [str(raw_ind)]

                                save_result = database.save_complete_scan(
                                    extracted_url=getattr(item, "raw_payload", None),
                                    qr_payload_type=getattr(item, "payload_type", "OTHER"),
                                    predicted_class=pred_class,
                                    risk_score=normalized_score,
                                    detected_indicators=indicators,
                                    selected_action=getattr(decision, "final_action", "WARN"),
                                    decision_reason=getattr(decision, "explanation", "Automated threat evaluation"),
                                    scan_id=client_scan_uuid,
                                    model_version=ml_status.get("metadata", {}).get("model_version", "v1.0.0") if ml_status else "v1.0.0",
                                    policy_version=getattr(decision, "policy_version", "v1.0.0"),
                                )
                                if save_result:
                                    active_scan_id = save_result["scan_id"]
                                    st.session_state.setdefault("saved_scan_tokens", set()).add(scan_token)
                                    st.session_state[f"scan_id_{scan_token}"] = active_scan_id

                            if active_scan_id:
                                fb_record = database.insert_rl_feedback(
                                    scan_id=active_scan_id,
                                    state_label=f"STATE_{decision.state_key}_OBS_{fb_gt}",
                                    selected_action=fb_sel_action,
                                    reward=fb_reward_val,
                                    feedback_source="USER",
                                )
                                if fb_record:
                                    st.success("✅ **Feedback Logged:** Recorded in Supabase `rl_feedback` table for offline policy analysis.")
                                else:
                                    st.error("❌ Failed to log feedback in database.")
                            else:
                                st.warning("⚠️ Could not establish a parent scan record before logging feedback.")

                # 6. Extracted Lexical & Structural Features (15 Canonical Metrics)
                st.markdown("#### 📊 Extracted URL Features (Scikit-Learn Ready)")

                with st.expander("🔍 View All 15 Canonical Features", expanded=False):
                    feat = analysis.features

                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        st.metric("URL Length", feat.get("url_length", 0))
                        st.metric("Hostname Length", feat.get("hostname_length", 0))
                        st.metric("Hostname Dots", feat.get("num_dots_hostname", 0))
                        st.metric("Subdomain Count", feat.get("num_subdomains", 0))
                        st.metric("Uses HTTPS", "Yes (1)" if feat.get("is_https", 0) == 1 else "No (0)")
                    with f_col2:
                        st.metric("Host is IP Address", "Yes (1)" if feat.get("is_ip_address", 0) == 1 else "No (0)")
                        st.metric("Contains '@' Symbol", "Yes (1)" if feat.get("has_at_symbol", 0) == 1 else "No (0)")
                        st.metric("Hostname Hyphens", feat.get("num_hyphens_hostname", 0))
                        st.metric("Hostname Digits", feat.get("num_digits_hostname", 0))
                        st.metric("Suspicious Keywords", feat.get("num_suspicious_keywords", 0))
                    with f_col3:
                        st.metric("Special Characters", feat.get("num_special_chars", 0))
                        st.metric("Path Length", feat.get("path_length", 0))
                        st.metric("Query Parameters", feat.get("num_query_params", 0))
                        st.metric("URL Encoding (%xx)", "Yes (1)" if feat.get("has_url_encoding", 0) == 1 else "No (0)")
                        st.metric("Unusual Scheme", "Yes (1)" if feat.get("has_unusual_scheme", 0) == 1 else "No (0)")

                    # Structured Table format
                    st.markdown("**Tabular Feature Vector:**")
                    df_features = pd.DataFrame([feat])
                    st.dataframe(df_features, use_container_width=True)

                # 6. Random Forest Model Evaluation Metrics
                if is_valid_url:
                    ml_status = get_classifier_status()
                    if ml_status["is_model_available"]:
                        with st.expander("📈 View Random Forest Evaluation Metrics & Diagnostics", expanded=False):
                            meta = ml_status.get("metadata", {})
                            eval_metrics = meta.get("evaluation_metrics", {})
                            d_info = meta.get("dataset_info", {})

                            st.markdown("##### 🧪 Performance on Held-Out Test Split")
                            st.caption(
                                f"Evaluated on a strictly held-out test split of **{d_info.get('test_samples', 30)} URLs** "
                                f"(Stratified: {eval_metrics.get('dataset_summary', {}).get('benign_samples', 15)} Benign, "
                                f"{eval_metrics.get('dataset_summary', {}).get('malicious_samples', 15)} Malicious) "
                                f"from {d_info.get('total_samples', 120)} total curated samples with zero train/test data leakage."
                            )

                            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                            with m_col1:
                                st.metric("Accuracy", f"{eval_metrics.get('accuracy', 0.0) * 100:.1f}%")
                            with m_col2:
                                st.metric("Precision", f"{eval_metrics.get('precision', 0.0) * 100:.1f}%")
                            with m_col3:
                                st.metric("Recall", f"{eval_metrics.get('recall', 0.0) * 100:.1f}%")
                            with m_col4:
                                st.metric("F1-Score", f"{eval_metrics.get('f1_score', 0.0):.3f}")
                            with m_col5:
                                roc = eval_metrics.get("roc_auc")
                                st.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "N/A")

                            # Plotly Visualizations
                            viz_col1, viz_col2 = st.columns(2)
                            with viz_col1:
                                cm_dict = eval_metrics.get("confusion_matrix", {})
                                if cm_dict:
                                    fig_cm = plot_confusion_matrix(cm_dict, title="Held-Out Test Confusion Matrix")
                                    st.plotly_chart(fig_cm, use_container_width=True)

                            with viz_col2:
                                feat_names = meta.get("feature_names", [])
                                importances = list(meta.get("feature_importances", {}).values())
                                if feat_names and importances:
                                    fig_imp = plot_feature_importance(
                                        feat_names,
                                        importances,
                                        top_n=8,
                                        title="Top Feature Importances (Gini)",
                                    )
                                    st.plotly_chart(fig_imp, use_container_width=True)

                # 7. Q-Learning Training Analytics & Policy Benchmark
                with st.expander("🧠 View Q-Learning Training Analytics & Policy Benchmark", expanded=True):
                    rl_agent = get_rl_agent()
                    rl_meta = rl_agent.training_metadata
                    tab_rl_train, tab_rl_bench = st.tabs(["📈 Training Rewards & Action Distribution", "⚖️ Baseline Comparison (Q-Learning vs Fixed-Threshold)"])

                    with tab_rl_train:
                        st.markdown("##### 🧪 Q-Learning Training Dynamics")
                        results_data = rl_meta.get("results", {})
                        hyper = rl_meta.get("hyperparameters", {})

                        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                        with r_col1:
                            st.metric("Total Episodes", rl_meta.get("episodes", 400))
                        with r_col2:
                            st.metric("Cumulative Reward", f"{results_data.get('total_cumulative_reward', 0.0):.1f}")
                        with r_col3:
                            st.metric("Learned Discrete States", results_data.get("total_learned_states", len(rl_agent.q_table)))
                        with r_col4:
                            st.metric("Final Exploration (ε)", f"{hyper.get('final_epsilon', 0.01):.3f}")

                        cum_rewards = rl_meta.get("cumulative_rewards", [])
                        action_dist = results_data.get("action_distribution", {})

                        c_col1, c_col2 = st.columns(2)
                        with c_col1:
                            if cum_rewards:
                                fig_reward = plot_cumulative_rewards(cum_rewards)
                                st.plotly_chart(fig_reward, use_container_width=True)
                        with c_col2:
                            if action_dist:
                                fig_dist = plot_action_distribution(action_dist)
                                st.plotly_chart(fig_dist, use_container_width=True)

                    with tab_rl_bench:
                        st.markdown("##### ⚖️ Comparative Evaluation on Held-Out Test Scenarios")
                        st.caption(
                            "Rigorous side-by-side comparison between the Adaptive Q-Learning policy (with safety constraints) "
                            "and a traditional fixed risk-threshold baseline (ALLOW < 30%, WARN 30-70%, BLOCK > 70%) "
                            "evaluated on identical test scenarios."
                        )

                        bench_results = prevention_engine.benchmark_policies()
                        fixed_p = bench_results.get("fixed_threshold_policy", {})
                        rl_p = bench_results.get("q_learning_adaptive_policy", {})

                        df_bench = pd.DataFrame([
                            {
                                "Evaluation Metric": "Malicious URL Allow Rate (False Negatives)",
                                "Fixed-Threshold Policy": f"{fixed_p.get('malicious_allow_rate', 0.0)}%",
                                "Adaptive Q-Learning Policy": f"{rl_p.get('malicious_allow_rate', 0.0)}%",
                                "Target / Direction": "Lower is Better (0% ideal)",
                            },
                            {
                                "Evaluation Metric": "Legitimate URL Block Rate (False Positives)",
                                "Fixed-Threshold Policy": f"{fixed_p.get('benign_block_rate', 0.0)}%",
                                "Adaptive Q-Learning Policy": f"{rl_p.get('benign_block_rate', 0.0)}%",
                                "Target / Direction": "Lower is Better (0% ideal)",
                            },
                            {
                                "Evaluation Metric": "Warning Rate (Prudent Caution)",
                                "Fixed-Threshold Policy": f"{fixed_p.get('warning_rate', 0.0)}%",
                                "Adaptive Q-Learning Policy": f"{rl_p.get('warning_rate', 0.0)}%",
                                "Target / Direction": "Balanced Friction",
                            },
                            {
                                "Evaluation Metric": "Optimal Decision Accuracy",
                                "Fixed-Threshold Policy": f"{fixed_p.get('optimal_decision_rate', 0.0)}%",
                                "Adaptive Q-Learning Policy": f"{rl_p.get('optimal_decision_rate', 0.0)}%",
                                "Target / Direction": "Higher is Better",
                            },
                            {
                                "Evaluation Metric": "Total Cumulative Reward / Utility",
                                "Fixed-Threshold Policy": f"{fixed_p.get('total_cumulative_reward', 0.0):.1f}",
                                "Adaptive Q-Learning Policy": f"{rl_p.get('total_cumulative_reward', 0.0):.1f}",
                                "Target / Direction": "Higher is Better",
                            },
                        ])
                        st.table(df_bench)

                        st.info(
                            "📌 **Experimental Analysis:** While the fixed-threshold baseline applies static "
                            "arbitrary cutoffs at 30% and 70%, the adaptive Q-Learning agent incorporates multi-signal "
                            "uncertainty and feature confidence to balance user friction against catastrophic breach risk."
                        )

                st.markdown("---")

# =============================================================================
# Tab 2: Supabase Scan History & Security Audit Log
# =============================================================================
with tab_history:
    st.markdown("### 📜 Supabase Scan History & Security Audit Log")
    st.caption("Centralized audit repository tracking decoded QR payloads, Random Forest classifications, and adaptive Q-Learning prevention decisions.")

    if not database.is_database_configured():
        st.warning("⚠️ **Supabase Credentials Not Configured in `.env`**")
        st.info(
            "Persistent scan history is stored securely in Supabase PostgreSQL. "
            "To activate persistent audit logging, configure `SUPABASE_URL` and `SUPABASE_KEY` in your project's `.env` file.\n\n"
            "**Setup Example (`.env`):**\n"
            "```bash\n"
            "SUPABASE_URL=https://your-project.supabase.co\n"
            "SUPABASE_KEY=your-anon-or-service-key\n"
            "```\n\n"
            "Once defined, restart Streamlit or refresh the application to view live database records."
        )
    else:
        is_conn, conn_msg = database.test_connection()
        if not is_conn:
            st.error(f"🔴 **Supabase Connection Issue:** {conn_msg}")
            if "schema.sql" in conn_msg:
                st.info("💡 **Database Setup Required:** Execute `supabase/schema.sql` in your Supabase SQL Editor to initialize the required tables.")
        else:
            # Refresh / Query Controls Header
            h_hdr1, h_hdr2 = st.columns([4, 1])
            with h_hdr1:
                st.markdown("Connected to live Supabase PostgreSQL instance. All scan events are logged immutably.")
            with h_hdr2:
                if st.button("🔄 Refresh History", key="btn_refresh_audit_log", use_container_width=True):
                    st.rerun()

            with st.spinner("Fetching scan audit logs from Supabase..."):
                detailed_scans = database.get_detailed_scan_history(limit=100)

            if not detailed_scans:
                st.info(
                    "ℹ️ **No Scan History Records Found**\n\n"
                    "Scan records will appear here once you analyze a QR code in the "
                    "**Real-Time QR Scanner** tab and click **Save Scan Record to Supabase**."
                )
            else:
                # Summary KPI metrics
                total_scans = len(detailed_scans)
                malicious_count = sum(1 for s in detailed_scans if s.get("predicted_class") == "MALICIOUS")
                warn_count = sum(1 for s in detailed_scans if s.get("prevention_action") in ("WARN", "WARN_USER"))
                block_count = sum(1 for s in detailed_scans if s.get("prevention_action") == "BLOCK")
                allow_count = sum(1 for s in detailed_scans if s.get("prevention_action") == "ALLOW")

                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                with kpi1:
                    st.metric("Total Scans Logged", total_scans)
                with kpi2:
                    st.metric("Malicious URLs", malicious_count)
                with kpi3:
                    st.metric("Blocks Enforced", block_count)
                with kpi4:
                    st.metric("Warnings Issued", warn_count)
                with kpi5:
                    st.metric("Clean Allowed", allow_count)

                st.markdown("---")

                # Search and Filtering Controls
                f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
                with f_col1:
                    search_query = st.text_input("🔍 Search URL / Payload Substring:", key="hist_search_query", placeholder="Filter by domain or keyword...")
                with f_col2:
                    cat_filter = st.selectbox("Predicted Category Filter:", ["ALL", "SAFE", "SUSPICIOUS", "MALICIOUS"], key="hist_cat_filter")
                with f_col3:
                    act_filter = st.selectbox("Action Filter:", ["ALL", "ALLOW", "WARN", "BLOCK"], key="hist_act_filter")

                # Filter records
                filtered_records = []
                for s in detailed_scans:
                    url_str = str(s.get("extracted_url", "")).lower()
                    if search_query and search_query.strip().lower() not in url_str:
                        continue
                    if cat_filter != "ALL" and s.get("predicted_class") != cat_filter:
                        continue
                    if act_filter != "ALL":
                        action_normalized = "WARN" if s.get("prevention_action") == "WARN_USER" else s.get("prevention_action")
                        if action_normalized != act_filter:
                            continue
                    filtered_records.append(s)

                st.caption(f"Displaying **{len(filtered_records)}** of **{total_scans}** logged scans")

                if filtered_records:
                    # Format DataFrame for presentation
                    df_display = pd.DataFrame([
                        {
                            "Timestamp": str(r.get("timestamp", ""))[:19].replace("T", " "),
                            "Payload Type": r.get("payload_type", "OTHER"),
                            "Extracted URL / Content": r.get("extracted_url", ""),
                            "ML Category": r.get("predicted_class", "N/A"),
                            "Risk Score": f"{float(r['risk_score']) * 100:.1f}%" if r.get("risk_score") is not None else "N/A",
                            "Enforced Action": "WARN" if r.get("prevention_action") == "WARN_USER" else r.get("prevention_action", "N/A"),
                            "Decision Reason": r.get("decision_reason", "N/A"),
                            "Scan ID": str(r.get("scan_id", ""))[:8] + "...",
                        }
                        for r in filtered_records
                    ])
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

                    # CSV Export
                    csv_data = database.export_scan_history_to_csv(filtered_records)
                    st.download_button(
                        label="📥 Export Filtered History to CSV",
                        data=csv_data,
                        file_name="qrshield_scan_history.csv",
                        mime="text/csv",
                        type="secondary",
                        help="Download RFC-4180 compliant CSV of filtered scan records."
                    )
                else:
                    st.warning("⚠️ No scan records match the current filter criteria.")
