# Speaker ID

Fast speaker identification service using **WeSpeaker CAMPPlus** (512-dim embeddings, VoxCeleb).  
Accepts Opus/WebRTC audio from mobile browsers — converts to 16kHz mono WAV, extracts embeddings, identifies by cosine similarity.

**Accuracy: 0.97** on Opus-compressed phone audio (vs. 0.16–0.32 with ECAPA).

---

## Requirements

- **Docker** with Compose v2
- **NVIDIA GPU** (Pascal+, tested on P104-100) with CUDA 11.8 drivers on the host
- `nvidia-container-toolkit` installed on the Docker host

---

## Quick Start

```bash
cd speaker_id_service
docker compose up --build
```

Wait for the log line:

```
CAMPPlus model successfully loaded!
```

Server runs on **http://localhost:8001**.

---

## Usage

### 1. Browser Enrollment

Open **http://localhost:8001/enroll** in any modern browser (Chrome, Safari, Firefox).

1. Enter a username (e.g. `alexander`)
2. Click **Записать** (Record) — read the shown text for ~5 seconds
3. Click **Стоп** (Stop) — review the recording
4. Repeat for 3 samples
5. Click **Зарегистрировать голос** (Register Voice)

Three samples are averaged into one `.npy` embedding file.

---

### 2. API Enrollment (Advanced)

```bash
curl -X POST http://localhost:8001/enroll \
  -F "user_id=alexander" \
  -F "files=@sample1.wav" \
  -F "files=@sample2.wav" \
  -F "files=@sample3.wav"
```

Response:
```json
{"status": "success", "user_id": "alexander"}
```

---

### 3. Identify Speaker

```bash
curl -X POST http://localhost:8001/identify \
  -F "file=@recording.ogg"
```

Response:
```json
{"user_id": "alexander", "confidence": 0.97}
```

If confidence is below **0.4**, returns `{"user_id": "unknown", "confidence": <score>}`.

---

## Audio Requirements

| Aspect | Requirement |
|--------|-------------|
| Format | Any (Opus, WebM, OGG, WAV, MP4, M4A — FFmpeg handles it) |
| Duration | ≥ 0.25s (4000 samples at 16kHz) |
| Channels | Auto-converted to mono |
| Sample rate | Auto-resampled to 16kHz |

Phone recordings via MediaRecorder API (`audio/webm; codecs=opus`, ~20 kbps) work well.

---

## How It Works

```
Raw Audio (Opus/WAV/WebM)
  │
  ▼ FFmpeg ──→ 16kHz mono WAV
  │
  ▼ Peak Normalization ──→ amplitude / peak × 0.9
  │
  ▼ torchaudio.compliance.kaldi ──→ 80-dim log Mel-fbank (25ms / 10ms)
  │
  ▼ CAMPPlus (7.2M params) ──→ 512-dim embedding
  │
  ▼ L2-normalize + cosine similarity against enrolled embeddings
  │
  ▼ { user_id, confidence }
```

### Why peak normalization?

Phone browsers record Opus at low volume (~20 kbps, amplitude ~0.22).  
Desktop WAV enrollment records at full volume (amplitude ~0.99).  
Without normalization, similarity scores drop from 0.97 to ~0.26.

---

## Project Structure

```
speaker-id/
├── README.md
├── DESIGN.md                    # Apple design system spec (enrollment UI)
├── setup_speaker_id.sh          # Legacy installer (kept for reference)
├── .gitignore
└── speaker_id_service/
    ├── docker-compose.yaml      # GPU passthrough, healthcheck, volumes
    ├── AGENTS.md                # Agent instructions for AI coding tools
    ├── enroll_client.sh         # CLI enrollment script (arecord + curl)
    └── speaker_id/
        ├── Dockerfile           # python:3.11-slim + CUDA 11.8
        ├── requirements.txt     # fastapi, uvicorn, torch, torchaudio, soundfile
        ├── .dockerignore
        ├── app.py               # FastAPI app (identify, enroll, browser UI)
        ├── campplus_model.py    # CAMPPlus architecture (WeSpeaker)
        └── pooling_layers.py    # TSTP/ASTP/ASP pooling layers
```

---

## Environment Variables

Configured in `docker-compose.yaml`:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN=0` | Suppress HuggingFace auth warnings |
| `HF_HUB_VERBOSITY=error` | Quiet download logs |

---

## Model

| Attribute | Value |
|-----------|-------|
| Architecture | CAMPPlus (Context-Aware Masking) |
| Source | WeSpeaker VoxCeleb |
| Embedding size | 512 |
| Parameters | 7.2M |
| Checkpoint | 63 MB (`campplus_avg_model.pt`) |
| Download | Auto-downloads from HuggingFace on first start |

---

## Developer

```bash
# View logs
docker compose logs -f

# Rebuild
docker compose up --build -d

# Stop
docker compose down

# Enter container
docker compose exec speaker_id bash
```
