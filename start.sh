#!/bin/bash

# start the FastAPI backend in the background
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# wait for the API to be ready before starting the dashboard
echo "Waiting for API to start..."
until curl -s http://localhost:8000/health > /dev/null; do
    sleep 2
done
echo "API is up"

# start Streamlit in the foreground
streamlit run src/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
