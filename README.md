[![SQL-ML Integration Test](https://github.com/webmaster-exit-1/SQL-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/webmaster-exit-1/SQL-ML/actions/workflows/ci.yml)

# 🛡️ SQL-ML: Distributed Error Detection for SQLMAP

SQL-ML is a high-performance detection engine that uses Machine Learning to classify SQL injection vulnerabilities. By utilizing Redis and a Producer-Consumer architecture, it allows sqlmap to perform scans at full speed while offloading complex structural analysis to background workers.

🏗️ Architecture
 * The Producer (redisql.py): A Flask API that receives HTTP logs from sqlmap and queues them in Redis.
 * The Broker (Redis): Manages the task queue, ensuring no data is lost during high-volume scans.
 * The Consumer (worker.py): Background workers that load the ML model once and process the queue.
 * The Pre-processor (newsql.py): Normalizes raw logs (e.g., id=105 → id=NUM) to improve model accuracy.
 * The Bridge (sqlmap_ml_bridge.py): Connects sqlmap directly to the API.

🚀 Getting Started

1. Prerequisites
 * Redis Server: `sudo apt install redis-server`
 * Python 3.8+
 * Dependencies:
   `pip install flask redis requests scikit-learn`

2. Installation
Clone the repository:
```bash
git clone https://github.com/webmaster-exit-1/SQL-ML.git
cd SQL-ML
```

3. Generate Training Data and Train Model
Before running the pipeline, you must generate the training data and train the ML model:

**Step 1: Generate Training Data**
Run the synthetic data generator to create `training_data.csv`:
```bash
python sdg.py
```
This script generates a dataset with labeled SQL injection examples and safe queries, which is required for training the machine learning model.

**Step 2: Train the Model**
Train the ML model using the generated training data:
```bash
python newsql.py
```
This creates the model files (`sql_model.pkl` and `vectorizer.pkl`) that the worker uses for detection.

4. Running the Pipeline
Use the provided automation script to start the services:
```bash
chmod +x start_pipeline.sh
./start_pipeline.sh
```
This starts Redis, the Flask API, and the ML Worker.

🛠️ Tool Integration
Integrating with SQLMAP

To analyze every request sqlmap makes in real-time, use the `--postprocess` flag:
```bash
python3 sqlmap.py -u "http://example.com/api.php?id=1" --postprocess=sqlmap_ml_bridge.py
```
Manual Check
You can also test a single log entry manually:
```bash
python3 ml_check.py "SELECT * FROM users WHERE id='1' UNION SELECT 1,2,3--"
```

📊 Monitoring
 * API Logs: `tail -f api.log`
 * ML Verdicts: `tail -f worker.log`
 * Redis Stats: `redis-cli monitor`

🧠 Machine Learning Strategy
Unlike standard regex-based detection, this tool focuses on:
 * Normalization: Stripping literals to focus on SQL structure.
 * Anomaly Detection: Identifying deviations from "clean" application responses.
 * Async Inference: Ensuring the ML overhead (100ms+) never blocks the scanner.

