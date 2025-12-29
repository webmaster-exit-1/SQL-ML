# 🛡️ SQL-ML: Distributed Error Detection for sqlmap

SQL-ML is a high-performance detection engine that uses Machine Learning to classify SQL injection vulnerabilities. By utilizing Redis and a Producer-Consumer architecture, it allows sqlmap to perform scans at full speed while offloading complex structural analysis to background workers.

🏗️ Architecture
 * The Producer (redisql.py): A Flask API that receives HTTP logs from sqlmap and queues them in Redis.
 * The Broker (Redis): Manages the task queue, ensuring no data is lost during high-volume scans.
 * The Consumer (worker.py): Background workers that load the ML model once and process the queue.
 * The Pre-processor (newsql.py): Normalizes raw logs (e.g., id=105 → id=NUM) to improve model accuracy.
 * The Bridge (sqlmap_ml_bridge.py): Connects sqlmap directly to the API.

🚀 Getting Started

1. Prerequisites
 * Redis Server: sudo apt install redis-server
 * Python 3.8+
 * Dependencies:
   pip install flask redis requests scikit-learn

2. Installation
Clone the repository and ensure your trained model file is in the root directory.

3. Running the Pipeline
Use the provided automation script to start the services:
`chmod +x start_pipeline.sh`
`./start_pipeline.sh`
This starts Redis, the Flask API, and the ML Worker.

🛠️ Tool Integration
Integrating with sqlmap

To analyze every request sqlmap makes in real-time, use the --postprocess flag:
`python3 sqlmap.py -u "http://example.com/api.php?id=1" --postprocess=sqlmap_ml_bridge.py`

Manual Check
You can also test a single log entry manually:
`python3 ml_check.py "SELECT * FROM users WHERE id='1' UNION SELECT 1,2,3--"`

📊 Monitoring
 * API Logs: `tail -f api.log`
 * ML Verdicts: `tail -f worker.log`
 * Redis Stats: `redis-cli monitor`

🧠 Machine Learning Strategy
Unlike standard regex-based detection, this tool focuses on:
 * Normalization: Stripping literals to focus on SQL structure.
 * Anomaly Detection: Identifying deviations from "clean" application responses.
 * Async Inference: Ensuring the ML overhead (100ms+) never blocks the scanner.

