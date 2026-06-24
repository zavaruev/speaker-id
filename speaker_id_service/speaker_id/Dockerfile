FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libasound2-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models/speaker_id /app/speakers

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["python3", "app.py"]
