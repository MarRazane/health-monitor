FROM python:3.11-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# create models directory (gets populated at runtime from HF Hub)
RUN mkdir -p models

# expose both ports
EXPOSE 8000 8501

# start both services via the entrypoint script
CMD ["bash", "start.sh"]
