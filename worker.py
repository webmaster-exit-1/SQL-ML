import redis
import json
import sys
from newsql import SQLMLDetector 

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
detector = SQLMLDetector()

def start_worker():
    try:
        detector.load_model()
        print("[*] Worker online. Waiting for sqlmap logs...", flush=True)
        print("-" * 50, flush=True)
    except Exception as e:
        print(f"[-] Critical Error: Could not load ML model: {e}", flush=True)
        sys.exit(1)

    while True:
        try:
            # BLPOP blocks until data arrives in the queue
            result_raw = r.blpop("sqlmap_logs", timeout=0)
            
            if result_raw:
                _, data = result_raw
                log_entry = json.loads(data)
                raw_log = log_entry.get('log', '')
                
                # Perform Inference
                # detector.model.predict_proba gives us the confidence percentage
                clean_text = detector.vectorizer.transform([raw_log])
                probs = detector.model.predict_proba(clean_text)[0]
                prediction = detector.model.predict(clean_text)[0]
                confidence = max(probs) * 100

                # Store result back in Redis for the API
                r.set(f"res:{log_entry['id']}", json.dumps(int(prediction)), ex=60)
                
                # FORMATTED LOGGING
                status = "🚩 MALICIOUS" if prediction == 1 else "✅ SAFE"
                
                # We truncate the log so it doesn't flood the terminal if it's huge
                log_preview = (raw_log[:75] + '..') if len(raw_log) > 75 else raw_log
                
                print(f"{status} ({confidence:.1f}%)", flush=True)
                print(f"  ID: {log_entry['id']}", flush=True)
                print(f"  RAW: {log_preview}", flush=True)
                print("-" * 50, flush=True)
                
        except Exception as e:
            print(f"[-] Worker Loop Error: {e}", flush=True)

if __name__ == "__main__":
    start_worker()
