#!/bin/bash

cd /app

# add /app to PYTHONPATH so 'src' is findable as a top-level package
export PYTHONPATH=/app

# start uvicorn by pointing it at the file directly, not the dotted module path
python -c "
import uvicorn
uvicorn.run('src.api.main:app', host='0.0.0.0', port=8000)
" &

echo "Waiting for API to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is up"
        break
    fi
    echo "  attempt $i/30..."
    sleep 3
done

python -m streamlit run src/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true