import requests
import time
import sys

API_URL = "http://localhost:5000"

def run_ml_tool(log_data):
    # 1. Send log to the Enqueue endpoint
    print(f"[*] Sending log to ML Pipeline...")
    response = requests.post(f"{API_URL}/process", json={"logs": [log_data]})
    
    if response.status_code != 200:
        print("[-] Error connecting to API")
        return

    job_id = response.json()['job_ids'][0]
    print(f"[*] Job Queued: {job_id}. Waiting for ML inference...")

    # 2. Poll for the result
    while True:
        res_node = requests.get(f"{API_URL}/results/{job_id}")
        if res_node.status_code == 200:
            data = res_node.json()['data']
            print("\n[!] ML ANALYSIS COMPLETE")
            verdict = "MALICIOUS" if data == 1 else "SAFE"
            print(f"Verdict: {verdict}")
            break
        elif res_node.status_code == 202:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(0.5)
        else:
            print("\n[-] Analysis failed.")
            break

if __name__ == "__main__":
    # Example usage: python ml_check.py "raw sql error log here"
    if len(sys.argv) > 1:
        run_ml_tool(sys.argv[1])
    else:
        print("Usage: python ml_check.py <log_string>")
