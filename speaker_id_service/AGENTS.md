# Speaker ID Service

## Architecture

- **Entrypoint**: `speaker_id/app.py` — FastAPI on port 8001, launched via `python3 app.py` (not uvicorn CLI)
- **Model**: `speechbrain/spkrec-ecapa-voxceleb` cached at `/app/models/speaker_id`
- **Storage**: enrolled speaker embeddings as `.npy` in `/app/speakers`
- **Audio**: FFmpeg auto-converts uploads to 16kHz mono WAV. No manual conversion needed.
- **Confidence threshold**: 0.4 — below this, response is `"unknown"`
- **GPU fallback**: on `RuntimeError` (e.g. cuFFT), encodes on CPU, then moves model back to GPU

## API

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/identify` | `file` (single UploadFile) | `{ user_id, confidence }` |
| GET | `/enroll` | — | HTML form |
| POST | `/enroll` | `user_id` (form) + `files` (multiple UploadFile) | `{ status, user_id }` |

Multiple audio samples are average-embedded into one `.npy`. Temporary files use UUID to survive concurrent requests.

## Developer Commands

```bash
# From speaker_id_service/
docker compose up --build
docker compose logs -f
docker compose down
```

GPU: NVIDIA with CUDA 11.8 (Pascal+). Container runs `nvidia` device driver reservation.

- No tests, no CI, no lint/typecheck config exist.

## Environment quirks

- `HF_TOKEN=0` and `HF_HUB_VERBOSITY=error` set in docker-compose to suppress HuggingFace noise
- `SB_LOG_LEVEL=ERROR`, `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, `HF_HUB_DISABLE_PROGRESS_BARS=1` set at app startup
- Minimum audio: 4000 samples (~0.25s at 16kHz)
