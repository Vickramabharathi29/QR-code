import os
import sys
import uuid
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import database
from modules.reinforcement_learning import compute_reward

def run_verification():
    print("=" * 60)
    print("QRShield Supabase Integration Verification")
    print("=" * 60)

    # 1. Check database connectivity
    configured = database.is_database_configured()
    print(f"Database Configured: {configured}")
    if not configured:
        print("ERROR: Supabase credentials not found!")
        sys.exit(1)

    client = database.get_supabase_client(require_service_role=True)
    success, msg = database.test_connection(client)
    print(f"Database Connection Test: {success} ({msg})")
    if not success:
        print(f"ERROR: Cannot connect to Supabase: {msg}")
        sys.exit(1)

    # Fetch an admin user id from user_roles if available
    admin_uid = None
    try:
        roles_res = client.table("user_roles").select("user_id, email, role").limit(1).execute()
        if roles_res.data:
            admin_uid = roles_res.data[0].get("user_id")
            print(f"Found existing user for test scans: {admin_uid} ({roles_res.data[0].get('email')})")
    except Exception as e:
        print(f"Note querying user_roles: {e}")

    # 2. Test Permitted Scan
    print("\n--- Testing Permitted (Safe) Scan ---")
    safe_url = "https://verified-safe.gov.sg/public-survey"
    safe_scan_id = str(uuid.uuid4())
    safe_class = "BENIGN"
    safe_score = 0.05
    safe_indicators = ["clean_tld", "https_enforced", "valid_ssl"]
    safe_action = "ALLOW"
    safe_reason = "Verified government domain with zero threat indicators."
    safe_state = "BENIGN_LOW_RISK"
    safe_reward = compute_reward(safe_class, safe_action)

    safe_res = database.save_complete_scan(
        extracted_url=safe_url,
        qr_payload_type="URL",
        predicted_class=safe_class,
        risk_score=safe_score,
        detected_indicators=safe_indicators,
        selected_action=safe_action,
        decision_reason=safe_reason,
        scan_id=safe_scan_id,
        user_id=admin_uid,
        destination_domain="verified-safe.gov.sg",
        model_version="rf-v2.1",
        policy_version="ql-v2.1",
        state_label=safe_state,
        reward=safe_reward,
        feedback_source="SYSTEM",
        client=client,
    )

    print("Safe scan persistence result:")
    for k, v in safe_res.items():
        if k != "errors":
            print(f"  {k}: {'OK' if v else 'FAIL'}")
    if safe_res.get("errors"):
        print(f"  Errors encountered: {safe_res['errors']}")

    # 3. Test Malicious Scan
    print("\n--- Testing Malicious QR Scan ---")
    mal_url = "http://192.168.1.105/credential-harvest/login.php?user=admin"
    mal_scan_id = str(uuid.uuid4())
    mal_class = "MALICIOUS"
    mal_score = 0.95
    mal_indicators = ["ip_address_host", "http_unencrypted", "credential_phishing_keywords"]
    mal_action = "BLOCK"
    mal_reason = "Raw IP address host with deceptive login page structure."
    mal_state = "CRITICAL_HIGH_RISK"
    mal_reward = compute_reward(mal_class, mal_action)

    mal_res = database.save_complete_scan(
        extracted_url=mal_url,
        qr_payload_type="URL",
        predicted_class=mal_class,
        risk_score=mal_score,
        detected_indicators=mal_indicators,
        selected_action=mal_action,
        decision_reason=mal_reason,
        scan_id=mal_scan_id,
        user_id=admin_uid,
        destination_domain="192.168.1.105",
        model_version="rf-v2.1",
        policy_version="ql-v2.1",
        state_label=mal_state,
        reward=mal_reward,
        feedback_source="SYSTEM",
        client=client,
    )

    print("Malicious scan persistence result:")
    for k, v in mal_res.items():
        if k != "errors":
            print(f"  {k}: {'OK' if v else 'FAIL'}")
    if mal_res.get("errors"):
        print(f"  Errors encountered: {mal_res['errors']}")

    # 4. Verify in each table individually
    print("\n--- Verifying Remote Supabase Tables ---")
    for name, sid in [("Safe Scan", safe_scan_id), ("Malicious Scan", mal_scan_id)]:
        print(f"\nChecking records for {name} (scan_id={sid}):")
        
        # scan_history
        sh = client.table("scan_history").select("*").eq("id", sid).execute()
        print(f"  scan_history: {len(sh.data)} row(s) found -> URL: {sh.data[0]['extracted_url'] if sh.data else 'NONE'}")

        # threat_analysis
        ta = client.table("threat_analysis").select("*").eq("scan_id", sid).execute()
        print(f"  threat_analysis: {len(ta.data)} row(s) found -> Class: {ta.data[0]['predicted_class'] if ta.data else 'NONE'}, Score: {ta.data[0]['risk_score'] if ta.data else 'NONE'}")

        # prevention_decisions
        pd = client.table("prevention_decisions").select("*").eq("scan_id", sid).execute()
        print(f"  prevention_decisions: {len(pd.data)} row(s) found -> Action: {pd.data[0]['selected_action'] if pd.data else 'NONE'}")

        # rl_feedback
        rl = client.table("rl_feedback").select("*").eq("scan_id", sid).execute()
        print(f"  rl_feedback: {len(rl.data)} row(s) found -> Action: {rl.data[0]['selected_action'] if rl.data else 'NONE'}, Reward: {rl.data[0]['reward'] if rl.data else 'NONE'}")

    # 5. Verify Admin History Joined Fetching
    print("\n--- Verifying Admin Dashboard Scan History Query ---")
    history = database.get_detailed_scan_history(limit=10, client=client, is_admin=True)
    print(f"Retrieved {len(history)} recent detailed scan records.")
    
    found_safe = False
    found_mal = False
    for item in history:
        if item.get("scan_id") == safe_scan_id:
            found_safe = True
            print(f"  [FOUND SAFE] ID={item['scan_id']} | URL={item['extracted_url']} | Verdict={item['predicted_class']} | Action={item['prevention_action']} | RL={item.get('rl_action')} (Reward: {item.get('rl_reward')}) | User={item.get('user_email')}")
        if item.get("scan_id") == mal_scan_id:
            found_mal = True
            print(f"  [FOUND MALICIOUS] ID={item['scan_id']} | URL={item['extracted_url']} | Verdict={item['predicted_class']} | Action={item['prevention_action']} | RL={item.get('rl_action')} (Reward: {item.get('rl_reward')}) | User={item.get('user_email')}")

    if found_safe and found_mal:
        print("\n>>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
    else:
        print(f"\nVerification incomplete: found_safe={found_safe}, found_mal={found_mal}")

if __name__ == "__main__":
    run_verification()
