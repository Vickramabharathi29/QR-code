"""QRShield: QR Scam and Malicious Redirection Detection.

Adaptive Threat Prevention Using Reinforcement Learning & Machine Learning.
Streamlit Web Application Entry Point.
"""

import streamlit as st
import config

st.set_page_config(
    page_title="QRShield | Threat Detection & Prevention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App Header
st.title("🛡️ QRShield")
st.subheader("QR Scam and Malicious Redirection Detection")
st.caption(
    "Using Machine Learning for Classification and Reinforcement Learning for Adaptive Threat Prevention"
)

# Sidebar System Health & Supabase Status
with st.sidebar:
    st.header("⚙️ System Status")
    
    from modules import database
    
    if not database.is_database_configured():
        st.warning("🟡 Supabase Not Configured (Set credentials in `.env`)")
        st.info("Refer to `supabase/schema.sql` to initialize your database.")
    else:
        connected, conn_msg = database.test_connection()
        if connected:
            st.success("🟢 Supabase Connected & Ready")
        elif "schema.sql" in conn_msg:
            st.warning("🟡 Supabase Connected (Tables Pending)")
            st.caption("Please run `supabase/schema.sql` in the Supabase SQL Editor.")
        else:
            st.error(f"🔴 Connection Failed: {conn_msg}")

    st.divider()
    st.markdown("**Core Technologies:**")
    st.markdown("- **QR Decoding:** OpenCV & Pyzbar")
    st.markdown("- **ML Classification:** Random Forest")
    st.markdown("- **Threat Prevention:** Q-Learning Agent")
    st.markdown("- **Database:** Supabase PostgreSQL")

# Main Content Placeholder
st.info(
    "👋 **Project Structure Initialized!**\n\n"
    "The core modules and Supabase schemas are set up. Next steps will implement the individual "
    "pipeline components: QR Decoding, URL & Redirection Analysis, Feature Extraction, "
    "Random Forest Classification, and the Reinforcement Learning Adaptive Engine."
)
