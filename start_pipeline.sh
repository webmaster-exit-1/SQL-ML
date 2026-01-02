#!/bin/bash

# --- Configuration ---
API_SCRIPT="redisql.py"
WORKER_SCRIPT="worker.py"
REDIS_PORT=6379

echo "[*] Initializing SQL-ML Detection Pipeline..."

# 1. Start Redis First (The Foundation)
if ! lsof -Pi :$REDIS_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "[!] Redis is not running. Starting redis-server..."
    redis-server --daemonize yes
    sleep 2
else
    echo "[+] Redis is healthy."
fi

# 2. Start the ML Worker SECOND (The Consumer)
# -u forces unbuffered output so 'tail -f worker.log' works in real-time
echo "[*] Launching ML Worker..."
python3 -u $WORKER_SCRIPT > worker.log 2>&1 &
WORKER_PID=$!

# 3. Start the API THIRD (The Producer)
echo "[*] Launching API Gateway..."
python3 -u $API_SCRIPT > api.log 2>&1 &
API_PID=$!

# 4. Success Check
sleep 2
if ps -p $WORKER_PID > /dev/null && ps -p $API_PID > /dev/null; then
    echo "------------------------------------------------"
    echo "[SUCCESS] Pipeline is LIVE in the correct order!"
    echo "    API PID: $API_PID (Logs: api.log)"
    echo "    Worker PID: $WORKER_PID (Logs: worker.log)"
    echo "------------------------------------------------"
    echo "Now run your tail commands and then sqlmap."
else
    echo "[!] Error: One or more processes failed to start. Check logs."
fi
