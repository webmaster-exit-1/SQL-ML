#!/bin/bash

# --- Configuration ---
API_SCRIPT="redisql.py"
WORKER_SCRIPT="worker.py"
REDIS_PORT=6379

echo "[*] Starting SQL-ML Detection Pipeline..."

# 1. Check if Redis is running
if ! lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "[!] Redis is not running. Starting redis-server..."
    redis-server --daemonize yes
    sleep 2
else
    echo "[+] Redis is already running."
fi

# 2. Start the Flask API in the background
echo "[*] Launching API Gateway ($API_SCRIPT)..."
python3 $API_SCRIPT > api.log 2>&1 &
API_PID=$!

# 3. Start the ML Worker in the background
echo "[*] Launching ML Worker ($WORKER_SCRIPT)..."
python3 $WORKER_SCRIPT > worker.log 2>&1 &
WORKER_PID=$!

echo "------------------------------------------------"
echo "[+] Pipeline is LIVE!"
echo "    API PID: $API_PID (Logs: api.log)"
echo "    Worker PID: $WORKER_PID (Logs: worker.log)"
echo "------------------------------------------------"
echo "Usage: python3 ml_check.py \"YOUR_SQL_LOG\""
echo "To stop the pipeline, run: kill $API_PID $WORKER_PID"
