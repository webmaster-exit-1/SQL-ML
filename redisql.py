import uuid
from flask import Flask, request, jsonify
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379)

@app.route('/process', methods=['POST'])
def enqueue_logs():
    logs = request.json.get('logs', [])
    job_ids = []

    for log in logs:
        job_id = str(uuid.uuid4())
        # Push to the list the workers are watching
        r.rpush("sqlmap_logs", json.dumps({"id": job_id, "log": log}))
        job_ids.append(job_id)

    return jsonify({"status": "queued", "job_ids": job_ids})

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    # Retrieve the result stored by the worker
    result = r.get(f"res:{job_id}")
    if result:
        return jsonify({"job_id": job_id, "status": "completed", "data": json.loads(result)})
    return jsonify({"job_id": job_id, "status": "pending"}), 202
