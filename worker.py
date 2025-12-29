import redis
import json
from newsql import SQLMLDetector # Update this line

r = redis.Redis(host='localhost', port=6379, db=0)
detector = SQLMLDetector()
detector.load_model() # Now loads the pkl files

def start_worker():
    print("Worker online. Listening for sqlmap logs...")
    while True:
        _, data = r.blpop("sqlmap_logs")
        log_entry = json.loads(data)
        
        # This now uses the trained model
        result = detector._process_single_log(log_entry['log'])
        
        r.set(f"res:{log_entry['id']}", json.dumps(result), ex=60)

if __name__ == "__main__":
    start_worker()

