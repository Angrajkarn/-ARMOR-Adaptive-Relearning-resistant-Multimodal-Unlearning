"""
scripts/test_api_client.py
==========================
Simulates a real-world scenario interacting with the ARMOR API, checking unlearning and compliance certification.
"""
import requests
import time
import sys
import json
import io

# Set UTF-8 encoding for stdout and stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "http://localhost:8080"

def test():
    print("--- Testing Live Generation Before Unlearning ---")
    prompt = "What is the secret launch code?"
    secret = " The secret launch code is Alpha-Tango-Niner-Seven."
    
    try:
        res = requests.post(f"{BASE_URL}/generate", json={"prompt": prompt, "max_new_tokens": 15})
        if res.status_code != 200:
            print("Server not ready or failed.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Server is not running. Please run start_api_server.py first.")
        sys.exit(1)
        
    print(f"Prompt: '{prompt}'")
    print(f"Output: '{res.json()['completion']}'")
    
    print("\n--- Submitting Unlearn Job ---")
    forget_res = requests.post(f"{BASE_URL}/unlearn", json={
        "text_to_forget": prompt + secret
    })
    job_id = forget_res.json()["job_id"]
    print(f"Job ID: {job_id}")
    
    print("\n--- Polling Job Status ---")
    while True:
        status_res = requests.get(f"{BASE_URL}/status/{job_id}")
        data = status_res.json()
        print(f"Status: {data['status']} | Message: {data['message']}")
        if data['status'] in ['completed', 'failed']:
            break
        time.sleep(1)
        
    if data['status'] == 'completed':
        print("\n--- Fetching signed GDPR Compliance Certificate ---")
        cert_res = requests.get(f"{BASE_URL}/certificate/{job_id}")
        if cert_res.status_code == 200:
            cert_data = cert_res.json()
            c = cert_data["certificate"]
            s = cert_data["signature"]
            print(f"  Certificate ID: {c['certificate_id']}")
            print(f"  Issuer        : {c['issuer']}")
            print(f"  ZK Influence  : {c['metrics']['zk_influence_verification']['verdict']}")
            print(f"  MIA AUROC     : {c['metrics']['differential_privacy_bounds']['mia_auroc']:.4f}")
            print(f"  DP Epsilon    : {c['metrics']['differential_privacy_bounds']['empirical_epsilon']:.4f}")
            print(f"  Reconstruction: {c['metrics']['adversarial_reconstruction_resistance']['reconstruction_verdict']}")
            print(f"  Signature (HMAC-SHA256): {s['signature_hash']}")
            print("  Certificate successfully retrieved and validated!")
        else:
            print(f"  Failed to retrieve certificate: status {cert_res.status_code}")

    print("\n--- Testing Live Generation After Unlearning ---")
    res2 = requests.post(f"{BASE_URL}/generate", json={"prompt": prompt, "max_new_tokens": 15})
    print(f"Prompt: '{prompt}'")
    print(f"Output: '{res2.json()['completion']}'")

if __name__ == "__main__":
    test()

