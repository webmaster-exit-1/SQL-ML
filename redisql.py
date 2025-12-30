import uuid
import json # <--- CRITICAL: Ensure this is here
from flask import Flask, request, jsonify
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.route('/process', methods=['POST'])
def enqueue_logs():
    logs = request.json.get('logs', [])
    job_ids = []
    for log in logs:
        job_id = str(uuid.uuid4())
        # The worker expects a JSON string
        r.rpush("sqlmap_logs", json.dumps({"id": job_id, "log": log}))
        job_ids.append(job_id)
    return jsonify({"status": "queued", "job_ids": job_ids})

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    result = r.get(f"res:{job_id}")
    if result:
        # result is already a JSON string from the worker ("MALICIOUS" or "SAFE")
        return jsonify({"job_id": job_id, "status": "completed", "data": json.loads(result)})
    return jsonify({"job_id": job_id, "status": "pending"}), 202

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
