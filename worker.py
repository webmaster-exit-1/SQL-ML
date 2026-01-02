import redis
import json
import sys
from newsql import SQLMLDetector 

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)
detector = SQLMLDetector()

def start_worker():
    try:
        detector.load_model()
        print("[*] Worker online. Waiting for sqlmap logs...", flush=True)
    except Exception as e:
        print(f"[-] Model Load Error: {e}", flush=True)
        sys.exit(1)

    while True:
        try:
            # blpop waits (blocks) until data is available in the queue
            result_raw = r.blpop("sqlmap_logs", timeout=0)
            if result_raw:
                _, data = result_raw
                log_entry = json.loads(data)
                
                # ML Inference
                result = detector._process_single_log(log_entry['log'])
                
                # Store result back in Redis for 60 seconds
                r.set(f"res:{log_entry['id']}", json.dumps(result), ex=60)
                
                # THIS IS THE MISSING PRINT THAT FILLS YOUR LOG
                print(f"[+] Processed ID: {log_entry['id']} | Result: {result}", flush=True)
        except Exception as e:
            print(f"[-] Worker Loop Error: {e}", flush=True)

if __name__ == "__main__":
    start_worker()
