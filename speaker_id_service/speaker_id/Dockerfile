# Base image: slim Python 3.11. CUDA comes from the host driver via
# nvidia-container-toolkit (see compose `deploy.resources.reservations`),
# not from this image.
FROM python:3.11-slim

# ffmpeg: universal audio decoding for uploads (Opus/WebM/OGG/MP4/M4A -> WAV)
# libsndfile1: native backend required by torchaudio/soundfile WAV loading.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned requirements first, then torch/torchaudio from the official
# cu118 wheel index (PyPI's default torch would pull a CPU-only or different
# CUDA build). Splitting the layers lets Docker cache the heavy installs:
# they only re-run when requirements.txt changes, not when app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test dependencies travel with the image so the suite is always runnable:
# `docker compose exec speaker_id python -m pytest tests -q` — no manual
# pip installs needed after a rebuild. Placed AFTER the multi-GB torch layer
# so it never invalidates that cache; only re-runs when requirements-dev.txt
# itself changes.
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Model + speakers dirs exist even on bare `docker run` without bind mounts;
# with compose they are overlaid by volumes anyway.
RUN mkdir -p /app/models/speaker_id /app/speakers

# App code last -> cheapest layer to invalidate on every code change.
COPY . .

# Unbuffered stdout so `docker compose logs -f` shows model-load progress live.
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

# uvicorn is started via app.py's __main__ block so TLS can be toggled through
# the SSL_KEYFILE / SSL_CERTFILE environment variables at run time.
CMD ["python3", "app.py"]
