"""Script to verify Supabase connectivity, live safe/malicious scan persistence,
and table integrity across scan_history, threat_analysis, prevention_decisions, and rl_feedback.
"""

import io
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import database

def run_verification() -> bool:
    print("=" * 70)
    print("QRSHIELD SUPABASE DATABASE & SCAN PERSISTENCE VERIFICATION")
    print("=" * 70)

    # 1. Connection & Diagnostic Verification
    print("\n[1/4] Verifying Supabase Connectivity & Diagnostics...")
    conn_ok, conn_msg = database.test_connection()
    print(f"  Connection Verified : {conn_ok}")
    print(f"  Connection Message  : {conn_msg}")
    if not conn_ok:
        print("❌ Supabase connection test failed!")
        return False

    diag = database.run_database_diagnostics()
    print(f"  Connection Status   : {diag.get('connection_status')}")
    print(f"  Write Status        : {diag.get('write_status')} ({diag.get('write_message')})")
    print(f"  Service Role Active : {diag.get('is_service_role')}")
    print(f"  Tables Accessible   : {', '.join(diag.get('tables_accessible', []))}")

    # 2. Permitted Safe QR Scan Execution
    print("\n[2/4] Executing Permitted Safe QR Scan Event...")
    safe_sid = str(uuid.uuid4())
    safe_url = "https://www.wikipedia.org"
    safe_res = database.save_complete_scan(
        extracted_url=safe_url,
        qr_payload_type="URL",
        predicted_class="SAFE",
        risk_score=0.06,
        detected_indicators=["HTTPS_ENCRYPTED", "BENIGN_REPUTATION"],
        selected_action="ALLOW",
        decision_reason="Verified benign domain via ML classifier and heuristic security rules.",
        scan_id=safe_sid,
        state_label="STATE:SAFE_RISK_6",
        reward=15.0,
        feedback_source="AUTOMATED",
    )
    if not safe_res:
        print(f"❌ Failed to persist permitted safe scan! Error: {database.get_last_error()}")
        return False
    print(f"  ✓ Permitted Safe Scan Persisted (Scan ID: {safe_sid})")

    # 3. Blocked Malicious QR Scan Execution
    print("\n[3/4] Executing Blocked Malicious QR Scan Event...")
    mal_sid = str(uuid.uuid4())
    mal_url = "http://192.168.1.100/secure-banking-login/verify-account.php"
    mal_res = database.save_complete_scan(
        extracted_url=mal_url,
        qr_payload_type="URL",
        predicted_class="MALICIOUS",
        risk_score=0.96,
        detected_indicators=["IP_HOST", "SUSPICIOUS_KEYWORD_LOGIN", "HTTP_UNENCRYPTED"],
        selected_action="BLOCK",
        decision_reason="Autonomous Zero-Trust Intercept: High-risk credential harvesting indicators detected.",
        scan_id=mal_sid,
        state_label="STATE:MALICIOUS_RISK_96",
        reward=25.0,
        feedback_source="AUTOMATED",
    )
    if not mal_res:
        print(f"❌ Failed to persist malicious scan! Error: {database.get_last_error()}")
        return False
    print(f"  ✓ Blocked Malicious Scan Persisted (Scan ID: {mal_sid})")

    # 4. Table Record Verification
    print("\n[4/4] Verifying Row Records in Supabase Tables...")
    client = database.get_supabase_client(require_service_role=True)
    if not client:
        print("❌ Could not get Supabase client for verification.")
        return False

    all_verified = True
    scans_to_verify = [
        ("PERMITTED (SAFE)", safe_sid, "SAFE", "ALLOW", 15.0),
        ("BLOCKED (MALICIOUS)", mal_sid, "MALICIOUS", "BLOCK", 25.0),
    ]

    for label, sid, exp_class, exp_action, exp_reward in scans_to_verify:
        print(f"\n  Checking Records for {label} [Scan ID: {sid}]:")
        
        # Table 1: scan_history
        sh_res = client.table("scan_history").select("*").eq("id", sid).execute().data
        if not sh_res:
            print(f"    ❌ scan_history: Record not found for {sid}")
            all_verified = False
        else:
            print(f"    ✓ scan_history: Found row | URL: {sh_res[0].get('extracted_url')[:35]}... | Status: {sh_res[0].get('scan_status')}")

        # Table 2: threat_analysis
        ta_res = client.table("threat_analysis").select("*").eq("scan_id", sid).execute().data
        if not ta_res:
            print(f"    ❌ threat_analysis: Record not found for {sid}")
            all_verified = False
        else:
            p_class = ta_res[0].get("predicted_class")
            r_score = ta_res[0].get("risk_score")
            print(f"    ✓ threat_analysis: Found row | Class: {p_class} (Expected: {exp_class}) | Risk Score: {r_score}")
            if p_class != exp_class:
                all_verified = False

        # Table 3: prevention_decisions
        pd_res = client.table("prevention_decisions").select("*").eq("scan_id", sid).execute().data
        if not pd_res:
            print(f"    ❌ prevention_decisions: Record not found for {sid}")
            all_verified = False
        else:
            act = pd_res[0].get("selected_action")
            print(f"    ✓ prevention_decisions: Found row | Action: {act} (Expected: {exp_action}) | Reason: {pd_res[0].get('decision_reason')[:40]}...")
            if act != exp_action:
                all_verified = False

        # Table 4: rl_feedback
        rl_res = client.table("rl_feedback").select("*").eq("scan_id", sid).execute().data
        if not rl_res:
            print(f"    ❌ rl_feedback: Record not found for {sid}")
            all_verified = False
        else:
            rew = rl_res[0].get("reward")
            print(f"    ✓ rl_feedback: Found row | Reward: {rew} (Expected: {exp_reward}) | Source: {rl_res[0].get('feedback_source')}")
            if rew != exp_reward:
                all_verified = False

    print("\n" + "=" * 70)
    if all_verified:
        print("✅ ALL 4 DATABASE TABLES AND SCAN EVENTS VERIFIED SUCCESSFULLY!")
    else:
        print("❌ Table verification had errors.")
    print("=" * 70)
    return all_verified

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
