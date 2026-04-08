FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /setup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup_models.py .
CMD ["python", "setup_models.py"]