FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ensure src is importable from anywhere
ENV PYTHONPATH=/app

RUN mkdir -p models && chmod +x start.sh

# create __init__.py files in case they weren't committed
RUN touch src/__init__.py src/api/__init__.py src/models/__init__.py src/data/__init__.py

EXPOSE 8000 8501

CMD ["bash", "start.sh"]