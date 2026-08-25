#!/bin/bash

echo "🚀 Начинаем развертывание Speaker ID с поддержкой CUDA..."

# 1. Создаем структуру папок
mkdir -p speaker_id_service/speaker_id
mkdir -p speaker_id_service/models/speaker_id
mkdir -p speaker_id_service/speakers

cd speaker_id_service

# 2. Генерируем requirements.txt
echo "📦 Создаем requirements.txt..."
cat << 'EOF' > speaker_id/requirements.txt
fastapi
uvicorn
numpy
python-multipart
pydantic
speechbrain
EOF

# 3. Генерируем полный app.py (на основе твоего кода)
echo "🐍 Создаем app.py..."
cat << 'EOF' > speaker_id/app.py
import os
import tempfile
import anyio
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path
import logging
import shutil
import uvicorn
import re

# Инициализация и логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _safe_remove(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.error(f"Failed to remove temp file {path}: {e}")

app = FastAPI()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limit

SPEAKERS_DIR = Path("/app/speakers")
MODELS_DIR = Path("/app/models/speaker_id")
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Проверка CUDA для P104-100
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"--- Speaker ID Service ---")
logger.info(f"Device: {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = os.environ.get("API_KEY", "default_secret_key")
    if api_key == expected_api_key:
        return api_key
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API Key",
    )

# Загружаем легковесную и точную модель от SpeechBrain
logger.info("Загрузка модели SpeechBrain...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODELS_DIR),
    run_opts={"device": device}
)
logger.info("Модель успешно загружена!")

class IdentifyResponse(BaseModel):
    user_id: str
    confidence: float

class EnrollResponse(BaseModel):
    status: str
    user_id: str

@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    """Распознавание спикера из аудиофайла."""
    filename = os.path.basename(file.filename) if file.filename else "upload"
    if not filename:
        filename = "upload"
    ext = os.path.splitext(filename)[1] or ".raw"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        temp_path = tmp.name
        
    try:
        file_size = 0
        async with await anyio.open_file(temp_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
                await buffer.write(chunk)

        signal, fs = await run_in_threadpool(torchaudio.load, temp_path)
        embeddings = classifier.encode_batch(signal)
        
        max_score = 0.0
        best_user = "unknown"
        
        for speaker_file in SPEAKERS_DIR.glob("*.npy"):
            enrolled_embedding = torch.tensor(np.load(speaker_file)).to(device)
            # Сравнение через косинусное сходство
            score = F.cosine_similarity(embeddings.squeeze(), enrolled_embedding.squeeze(), dim=0).item()
            if score > max_score:
                max_score = score
                best_user = speaker_file.stem
        
        # Проверка порога точности
        if max_score < 0.25:
            best_user = "unknown"
            
        return IdentifyResponse(user_id=best_user, confidence=max_score)
    finally:
        if os.path.exists(temp_path):
            await run_in_threadpool(_safe_remove, temp_path)

@app.post("/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), file: UploadFile = File(...), api_key: str = Security(get_api_key)):
    """Регистрация нового голоса (создание слепка .npy)"""
    safe_user_id = os.path.basename(user_id)
    if not safe_user_id or not re.match(r"^[a-zA-Z0-9_-]+$", safe_user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")

    filename = os.path.basename(file.filename) if file.filename else "upload"
    if not filename:
        filename = "upload"
    ext = os.path.splitext(filename)[1] or ".raw"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        temp_path = tmp.name
        
    try:
        file_size = 0
        async with await anyio.open_file(temp_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
                await buffer.write(chunk)

        signal, fs = await run_in_threadpool(torchaudio.load, temp_path)
        embeddings = classifier.encode_batch(signal)
        
        np.save(SPEAKERS_DIR / f"{safe_user_id}.npy", embeddings.squeeze().cpu().numpy())
        return EnrollResponse(status="success", user_id=safe_user_id)
    finally:
        if os.path.exists(temp_path):
            await run_in_threadpool(_safe_remove, temp_path)

if __name__ == "__main__":
    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    if ssl_keyfile and ssl_certfile:
        logger.info("Starting server with SSL/TLS enabled.")
        uvicorn.run(app, host="0.0.0.0", port=8001, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        logger.warning("WARNING: Starting server without SSL/TLS. In production, ensure this service is behind a reverse proxy (like Nginx) that handles HTTPS.")
        uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

# 4. Генерируем правильный Dockerfile с CUDA 11.8
echo "🐳 Создаем Dockerfile..."
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

# Принудительно ставим PyTorch с CUDA 11.8 для архитектуры Pascal (P104-100)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models/speaker_id /app/speakers

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["python3", "app.py"]
EOF

# 5. Генерируем docker-compose.yaml с пробросом GPU
echo "⚙️ Создаем docker-compose.yaml..."
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

# 6. Запускаем сборку и поднятие контейнера
echo "🔥 Запускаем сборку Docker..."
docker compose up -d --build

echo "✅ Готово! Контейнер запущен."
echo "Посмотреть логи можно командой: docker logs -f speaker_id_service"
