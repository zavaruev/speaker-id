#!/bin/bash

echo "Speaker ID deployment started..."

# 1. Create directory structure
mkdir -p speaker_id_service/speaker_id
mkdir -p speaker_id_service/models/speaker_id
mkdir -p speaker_id_service/speakers

cd speaker_id_service

# 2. Generate requirements.txt
echo "Creating requirements.txt..."
cat << 'EOF' > speaker_id/requirements.txt
fastapi
uvicorn
numpy
python-multipart
pydantic
speechbrain
EOF

echo "Creating app.py..."
cat << 'EOF' > speaker_id/app.py
import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path
import logging
import shutil
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

SPEAKERS_DIR = Path("/app/speakers")
MODELS_DIR = Path("/app/models/speaker_id")
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"--- Speaker ID Service ---")
logger.info(f"Device: {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

logger.info("Loading SpeechBrain model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODELS_DIR),
    run_opts={"device": device}
)
logger.info("Model loaded!")

class IdentifyResponse(BaseModel):
    user_id: str
    confidence: float

class EnrollResponse(BaseModel):
    status: str
    user_id: str

@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        signal, fs = torchaudio.load(temp_path)
        embeddings = classifier.encode_batch(signal)
        max_score = 0.0
        best_user = "unknown"
        for speaker_file in SPEAKERS_DIR.glob("*.npy"):
            enrolled_embedding = torch.tensor(np.load(speaker_file)).to(device)
            score = F.cosine_similarity(embeddings.squeeze(), enrolled_embedding.squeeze(), dim=0).item()
            if score > max_score:
                max_score = score
                best_user = speaker_file.stem
        if max_score < 0.25:
            best_user = "unknown"
        return IdentifyResponse(user_id=best_user, confidence=max_score)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        signal, fs = torchaudio.load(temp_path)
        embeddings = classifier.encode_batch(signal)
        np.save(SPEAKERS_DIR / f"{user_id}.npy", embeddings.squeeze().cpu().numpy())
        return EnrollResponse(status="success", user_id=user_id)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

echo "Creating Dockerfile..."
cat << 'EOF' > speaker_id/Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libasound2-dev \
    libsndfile1 \
    gcc \
    g++ \
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
EOF

echo "Creating docker-compose.yaml..."
cat << 'EOF' > docker-compose.yaml
version: '3.8'

services:
  speaker_id:
    build:
      context: ./speaker_id
      dockerfile: Dockerfile
    container_name: speaker_id_service
    ports:
      - "8001:8001"
    volumes:
      - ./models/speaker_id:/app/models/speaker_id
      - ./speakers:/app/speakers
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
EOF

echo "Building Docker container..."
docker compose up -d --build

echo "Done! Container started."
echo "Check logs: docker logs -f speaker_id_service"
