# Speaker ID

Speaker identification service with FastAPI + WeSpeaker CAMP++.

Accepts Opus/WebRTC audio from mobile phones, converts to 16kHz mono WAV, extracts 512-dim speaker embeddings via CAMPPlus (VoxCeleb), and performs cosine-similarity identification. GPU-ready with Docker Compose + CUDA 11.8.

Accuracy: **0.97** on Opus-compressed phone audio (vs 0.16–0.32 with SpeechBrain ECAPA).

## Quick Start

```bash
cd speaker_id_service
docker compose up --build
```

- `POST /identify` — upload audio, get `{ user_id, confidence }`
- `GET /enroll` — browser enrollment form (3 recordings)
- `POST /enroll` — upload samples, register voice

## Architecture

- **Model**: WeSpeaker CAMPPlus (512-dim, 7.2M params)
- **Audio pipeline**: FFmpeg → WAV → peak normalization → 80-dim Mel fbank → CAM++ → embedding
- **Backend**: FastAPI (port 8001)
- **GPU**: NVIDIA P104-100 with CUDA 11.8
