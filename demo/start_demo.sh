#!/usr/bin/env bash
# quarq demo launcher
# Starts quarq API server and opens Open WebUI in the browser.
# Usage: ./demo/start_demo.sh

set -e

echo "Starting quarq API server..."
quarq serve &
QUARQ_PID=$!
sleep 2

echo "quarq running at http://127.0.0.1:8000"
echo "Opening Open WebUI..."
open http://localhost:8080

echo ""
echo "Demo ready."
echo "  quarq API:    http://127.0.0.1:8000/docs"
echo "  Open WebUI:   http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop quarq."

wait $QUARQ_PID
