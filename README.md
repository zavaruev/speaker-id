# Speaker ID

Fast speaker identification service using **WeSpeaker CAMPPlus** (512-dim embeddings, VoxCeleb).
Accepts Opus/WebRTC audio from mobile browsers — converts to 16 kHz mono WAV, extracts embeddings, identifies by cosine similarity.

**Accuracy: 0.97** on Opus-compressed phone audio (vs. 0.16–0.32 with ECAPA).

---

## Requirements

- **Docker** with Compose v2
- **NVIDIA GPU** (Pascal+, tested on P104-100) with CUDA 11.8 drivers on the host
- `nvidia-container-toolkit` installed on the Docker host

---

## Setup: API Key (required)

Enrollment is protected by an API key. Since the latest version, `docker-compose.yaml`
reads it from a `.env` file next to `docker-compose.yaml` and **refuses to start without it**
(`API_KEY_missing_in_dotenv`):

```bash
cd speaker_id_service
echo "API_KEY=change-me-to-a-long-random-string" > .env   # keep this secret
```

> `/identify` is intentionally unauthenticated (LAN/smart-home use). Put the service behind
> a reverse proxy if you expose it beyond a trusted network.

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

Server runs on **http://localhost:8001**. On the first start the 63 MB CAMPPlus checkpoint is
downloaded from HuggingFace into `./models/speaker_id/` and verified against a SHA-256 checksum.

---

## Usage: three ways to enroll

### 1. Browser Enrollment UI

Open **http://localhost:8001/enroll** in any modern browser (Chrome, Safari, Firefox).

1. Enter a username (e.g. `alexander`) — allowed characters: `a-z A-Z 0-9 _ -`
2. Enter your **API Key** (the value from `.env`)
3. Click **Record** — read the shown text for ~5 seconds
4. Click **Stop** — review the recording
5. Repeat for 3 samples
6. Click **Register Voice**

Samples are averaged into one `.npy` embedding file stored in `./speakers/`.

### 2. CLI Enrollment Client (`enroll_client.sh`)

Interactive shell client for Linux machines with a microphone (`arecord`). Records 3 × 8 s
samples and enrolls them. It reads the API key from the environment (`SPEAKER_ID_API_KEY`)
or asks for it interactively — older versions did not send the key and always got `401`.

```bash
export SPEAKER_ID_API_KEY="change-me-to-a-long-random-string"
export SPEAKER_ID_SERVER="http://192.168.22.102:8001"   # optional, this is the default target
./enroll_client.sh
```

### 3. Raw API Enrollment (curl)

```bash
curl -X POST http://localhost:8001/enroll \
  -H "X-API-Key: $API_KEY" \
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

## Identify a Speaker

```bash
curl -X POST http://localhost:8001/identify \
  -F "file=@recording.ogg"
```

Response:
```json
{"user_id": "alexander", "confidence": 0.97}
```

If confidence is below **0.4**, returns `{"user_id": "unknown", "confidence": <score>}`.
No API key required for identification.

---

## API Reference

| Method | Path | Auth | Input | Output |
|--------|------|------|-------|--------|
| POST | `/identify` | — | `file` (single audio UploadFile) | `{ user_id, confidence }` |
| GET | `/enroll` | — | — | HTML enrollment form |
| POST | `/enroll` | `X-API-Key` | `user_id` (form) + `files` (multiple UploadFile) | `{ status, user_id }` |
| GET | `/health` | — | — | `{"status": "ok"}` or `503` until the model is loaded |

Validation rules enforced server-side (HTTP 400 otherwise): `user_id` must match
`^[a-zA-Z0-9_-]+$`; filenames are reduced to their basename; max **50 files** per enrollment;
each file ≤ **50 MB** (HTTP 413); audio must be ≥ 4000 samples ≈ 0.25 s.

---

## Audio Requirements

| Aspect | Requirement |
|--------|-------------|
| Format | Any (Opus, WebM, OGG, WAV, MP4, M4A — FFmpeg handles it) |
| Duration | ≥ 0.25 s (4000 samples at 16 kHz) |
| Channels | Auto-converted to mono |
| Sample rate | Auto-resampled to 16 kHz |

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

### Storage & concurrency

- Enrolled voices live as `.npy` files in the bind-mounted `./speakers/` volume.
- Writes are atomic (temp file inside the volume + `os.replace`); an in-memory matrix of all
  embeddings is rebuilt after every enrollment so `/identify` does batched cosine similarity
  in one matmul.

---

## Project Structure

```
speaker-id/
├── README.md                       # This file
├── DESIGN.md                       # Apple design-system spec for the /enroll UI
├── setup_speaker_id.sh             # Legacy installer (kept for reference, do not use)
├── AGENTS.md                       # Repo-level notes for AI coding tools
└── speaker_id_service/
    ├── docker-compose.yaml         # GPU passthrough, API_KEY from .env, healthcheck
    ├── .env                        # Your secrets (gitignored): API_KEY=...
    ├── AGENTS.md                   # Service architecture notes for AI coding tools
    ├── enroll_client.sh            # CLI enrollment client (arecord + curl, sends X-API-Key)
    ├── models/speaker_id/          # CAMPPlus checkpoint (auto-downloaded, gitignored)
    ├── speakers/                   # Enrolled *.npy voice profiles (gitignored)
    └── speaker_id/
        ├── Dockerfile              # python:3.11-slim + ffmpeg + torch cu118
        ├── requirements.txt        # fastapi, uvicorn, numpy, python-multipart, pydantic, soundfile
        ├── app.py                  # FastAPI app: /identify, /enroll, inline browser UI
        ├── campplus_model.py       # CAMPPlus architecture (from WeSpeaker, Apache-2.0)
        ├── pooling_layers.py       # TSTP/ASTP/ASP/MHASTP pooling layers (from WeSpeaker)
        ├── benchmark*.py           # Event-loop blocking vs threadpool micro-benchmarks
        └── tests/                  # pytest suite (see Testing below)
```

---

## Environment Variables

Set in `.env` next to `docker-compose.yaml` / passed through compose:

| Variable | Required | Purpose |
|----------|----------|---------|
| `API_KEY` | **yes** (compose fails without it) | Shared secret for `POST /enroll`, checked against the `X-API-Key` header |
| `HF_TOKEN` | no | Set to `0` to suppress HuggingFace auth warnings |
| `HF_HUB_VERBOSITY` | no | Set to `error` to quiet download logs |
| `SSL_KEYFILE` / `SSL_CERTFILE` | no | Opt-in HTTPS for uvicorn; without them the server logs a warning and serves plain HTTP |

Client-side variables for `enroll_client.sh`: `SPEAKER_ID_API_KEY` (key),
`SPEAKER_ID_SERVER` (default `http://192.168.22.102:8001`).

---

## Model

| Attribute | Value |
|-----------|-------|
| Architecture | CAMPPlus (CAM++, Context-Aware Masking) |
| Source | WeSpeaker VoxCeleb |
| Embedding size | 512 |
| Parameters | 7.2M |
| Checkpoint | 63 MB (`campplus_avg_model.pt`), SHA-256 verified after download |
| Download | Auto-downloads from HuggingFace on first start |

---

## Testing

The suite lives in `speaker_id_service/speaker_id/tests/`. pytest is not part of the image,
so install it into the running container once and run:

```bash
cd speaker_id_service
docker compose exec speaker_id pip install -q pytest pytest-asyncio httpx
docker compose exec speaker_id python -m pytest tests -q
```

App/security/path-traversal tests mock out `torch`/`torchaudio`; model and pooling-layer tests
need the real torch stack and therefore run inside the container (or on a CUDA-capable host).

---

## Developer Commands

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

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Compose exits with `API_KEY_missing_in_dotenv` | Create `speaker_id_service/.env` containing `API_KEY=...` (see Setup) |
| `POST /enroll` returns 401 | Missing or wrong `X-API-Key` header; browser UI asks for the key, `enroll_client.sh` reads `SPEAKER_ID_API_KEY` |
| Container restart-loops, no GPU in logs | `nvidia-container-toolkit` missing or drivers not CUDA 11.8-compatible; service still works on CPU (slow) via the automatic CPU fallback |
| `Audio too short or empty` | Recording under 0.25 s or a corrupted upload |
