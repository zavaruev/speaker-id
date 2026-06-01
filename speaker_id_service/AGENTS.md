# Speaker ID Service

## Architecture

- **Entrypoint**: `speaker_id/app.py` — FastAPI on port 8001, launched via `python3 app.py`
- **Model**: WeSpeaker CAMPPlus (`/app/models/speaker_id/campplus_avg_model.pt`, 63 MB) — 512-dim embeddings
- **Model code**: `speaker_id/campplus_model.py` + `speaker_id/pooling_layers.py` (copied from WeSpeaker repo, no pip dep)
- **Storage**: enrolled speaker embeddings as `.npy` in `/app/speakers`
- **Audio pipeline**: 
  1. FFmpeg converts upload → 16kHz mono WAV
  2. Peak normalization (`signal / peak * 0.9`) — critical for Opus-compressed phone audio
  3. `torchaudio.compliance.kaldi` → 80-dim log Mel-fbank (25ms frame, 10ms shift)
  4. CAMPPlus → 512-dim embedding
- **Confidence threshold**: 0.4 — below this, response is `"unknown"`
- **GPU fallback**: on RuntimeError, encodes on CPU then moves model back to GPU

## Key Metrics

- ECAPA (old): 192-dim, accuracy ~0.16–0.32 on Opus phone audio
- **CAM++ (current)**: 512-dim, accuracy **0.97** on Opus phone audio

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

## Env vars (docker-compose.yaml)

- `HF_TOKEN=0`, `HF_HUB_VERBOSITY=error` — suppress HuggingFace noise (not used directly)
- `SB_LOG_LEVEL=ERROR`, `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, `HF_HUB_DISABLE_PROGRESS_BARS=1`
- Minimum audio: 4000 samples (~0.25s at 16kHz)
