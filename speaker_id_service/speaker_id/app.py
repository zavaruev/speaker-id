"""Speaker Identification HTTP service.

Exposes a FastAPI application on port 8001 backed by a WeSpeaker CAMPPlus
(CAM++) speaker-embedding model:

* ``POST /identify`` — upload any audio file, get the best-matching enrolled
  speaker and its cosine-similarity confidence (unauthenticated by design,
  meant for a trusted LAN / smart-home network).
* ``GET  /enroll``   — self-contained HTML enrollment UI (Apple-style design;
  spec lives in ../../DESIGN.md). The markup below is intentionally inline.
* ``POST /enroll``   — multi-file enrollment endpoint protected by an
  ``X-API-Key`` header; averages all samples into one L2-normalized embedding
  stored as ``/app/speakers/<user_id>.npy``.
* ``GET  /health``   — readiness probe used by the Docker healthcheck.

Concurrency model: every blocking operation in a request path (file writes,
``torchaudio.load``, temp-file cleanup, embedding persistence) is either
async or dispatched through ``run_in_threadpool`` so the event loop is never
blocked by disk I/O. GPU inference itself runs synchronously, which serializes
requests — acceptable for this single-user smart-home service.
"""

# HuggingFace environment variables MUST be set before torch/torchaudio pull in
# the huggingface_hub machinery, otherwise warnings are emitted at import time.
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import uuid
import sys
import asyncio
import tempfile
import anyio
import hashlib
sys.path.insert(0, "/app")
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from campplus_model import CAMPPlus
from pathlib import Path
import logging
import uvicorn
import threading
import re

# ---------------------------------------------------------------------------
# Logging setup: keep our own logs at INFO but silence chatty third-party
# loggers (httpx/urllib3/filelock) that would otherwise spam on every request.
# NOTE: _safe_remove below references `logger` at call time (not import time),
# so it is safe that it is textually defined before this line.
logging.basicConfig(level=logging.INFO)

def _safe_remove(path: str):
    """Best-effort synchronous file removal.

    Never raises: missing files are ignored, anything else is only logged.
    Always call via run_in_threadpool from async code — os.remove blocks.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")

for _logger in ["httpx", "urllib3", "filelock"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

# Bind-mounted volumes (see docker-compose.yaml). Both are created on the host
# side by compose; mkdir here also supports bare `python3 app.py` runs.
SPEAKERS_DIR = Path("/app/speakers")          # enrolled voice profiles (*.npy)
MODELS_DIR = Path("/app/models/speaker_id")   # CAMPPlus checkpoint cache
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model bootstrap (runs once at import time, before uvicorn starts serving).
# Prefer CUDA (tested on P104-100 / Pascal); transparently fall back to CPU.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"--- Speaker ID Service ---")
logger.info(f"Device: {device}")
if device != "cpu":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Download the WeSpeaker CAMPPlus checkpoint on first start and verify its
# integrity. The SHA-256 check only guards the download path — an existing
# checkpoint is assumed intact (it was verified when it was downloaded).
logger.info("Loading CAMPPlus model...")
model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
ckpt_path = MODELS_DIR / "campplus_avg_model.pt"
if not ckpt_path.exists():
    # First start: fetch the official VoxCeleb checkpoint (63 MB) from HF Hub.
    import urllib.request
    url = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/avg_model.pt"
    logger.info(f"Downloading CAM++ from {url}")
    urllib.request.urlretrieve(url, str(ckpt_path))

    expected_checksum = "07abeeb5150441995b51ea65c9ccc8feed78b33040012f1d2fad29a0e4f5b8d7"
    sha256_hash = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    if sha256_hash.hexdigest() != expected_checksum:
        os.remove(ckpt_path)
        logger.error("Model checksum verification failed.")
        raise RuntimeError("Model checksum verification failed.")

# Strip DataParallel's 'module.' prefix and drop the optional classification
# projection head — we only need the embedding backbone.
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
state_dict = {}
for k, v in ckpt.items():
    k = k.replace("module.", "")
    if not k.startswith("projection"):
        state_dict[k] = v
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # inference-only: disable dropout / BatchNorm running stats
logger.info("CAMPPlus model successfully loaded!")

# Warm-up: run dummy inference to compile CUDA kernels and allocate cuDNN
# workspaces so the first real request does not pay the latency.
with torch.no_grad():
    dummy = torch.randn(1, 100, 80).to(device)
    _ = model(dummy)
logger.info("Model warm-up done")
_model_ready = True  # flipped after warm-up; /health reports 503 until then

MAX_FILE_SIZE = 50 * 1024 * 1024  # per-file upload cap -> HTTP 413 above this
MAX_FILES = 50                    # max audio samples accepted per enrollment

# ---------------------------------------------------------------------------
# In-memory speaker gallery. All enrolled embeddings are stacked into one
# matrix so /identify computes cosine similarity against EVERY speaker in a
# single matmul instead of N separate dot products. Guarded by a lock because
# enrollment rebuilds it while identify reads it.
_embedding_names: list[str] = []                 # index i  <-> speaker name
_embedding_matrix: torch.Tensor | None = None    # (num_speakers, 512), L2-norm rows
_cache_lock = threading.Lock()
# Inference lock to prevent concurrent CPU fallbacks from moving the model
# while other threads are using it on the GPU.
_model_lock = threading.Lock()
# Pre-created no-op resampler (16k->16k); compute_fbank swaps it lazily only
# when a non-16kHz signal shows up, avoiding per-request transform init.
_resampler_16k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000).to(device)

API_KEY_NAME = "X-API-Key"
# auto_error=False so we can return a proper 401 JSON ourselves instead of
# FastAPI's default 403 when the header is missing.
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """FastAPI security dependency guarding POST /enroll.

    Compares the X-API-Key header against the API_KEY env var (injected by
    docker-compose from .env). If API_KEY is unset (bare dev runs only —
    compose enforces it), every request is rejected rather than falling back
    to an open door.
    """
    expected_api_key = os.environ.get("API_KEY")
    if not expected_api_key or api_key != expected_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )
    return api_key

def _model_inference(fbank_tensor: torch.Tensor) -> torch.Tensor:
    """Helper to run model inference (with CPU fallback) in a threadpool.

    Model inference is a heavy synchronous operation that blocks the asyncio
    event loop. Encapsulating it allows us to run it via run_in_threadpool.
    """
    with torch.no_grad():
        try:
            with _model_lock:
                # Ensure the model is on the correct device if it was temporarily moved by another thread
                if str(next(model.parameters()).device) != str(device):
                    model.to(device)
            embedding = model(fbank_tensor)
        except RuntimeError as e:
            with _model_lock:
                logger.warning(f"GPU inference failed, falling back to CPU: {e}")
                fbank_cpu = fbank_tensor.cpu()
                model_cpu = model.cpu()
                with torch.no_grad():
                    embedding = model_cpu(fbank_cpu)
                model.to(device)
                embedding = embedding.to(device)
        return embedding

def compute_fbank(signal: torch.Tensor, fs: int) -> torch.Tensor:
    """Waveform -> CAMPPlus input features.

    Resamples to 16 kHz if needed (lazily reconfiguring the shared global
    resampler), extracts Kaldi-compatible 80-dim log Mel filterbanks using
    the exact frame geometry CAMPPlus was trained with (25 ms window /
    10 ms shift), then mean-normalizes over time.

    Note on dither=1.0: this follows the classic Kaldi convention where
    waveforms are int16-scaled (dither noise ~ -90 dBFS). Our waveform is a
    peak-normalized float in [-1, 1], so torchaudio adds Gaussian noise of
    std ~1.0 directly onto it. Embeddings stay usable (speaker structure
    survives) but become slightly non-deterministic run-to-run (~0.05 cosine
    jitter measured on identical input).
    """
    if fs != 16000:
        global _resampler_16k
        if _resampler_16k.orig_freq != fs or _resampler_16k.new_freq != 16000:
            _resampler_16k = torchaudio.transforms.Resample(fs, 16000).to(signal.device)
        signal = _resampler_16k(signal)
        fs = 16000
    fbank = kaldi.fbank(signal, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=fs)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    return fbank.unsqueeze(0)  # (1, num_frames, feat_dim)

class IdentifyResponse(BaseModel):
    """Body of POST /identify: best match plus its cosine similarity."""
    user_id: str
    confidence: float

class EnrollResponse(BaseModel):
    """Body of POST /enroll on success."""
    status: str
    user_id: str

async def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio to 16 kHz mono WAV via FFmpeg.

    Runs ffmpeg as a true async subprocess (never blocks the event loop) and
    swallows its noisy stdout/stderr. Returns False on a non-zero exit code or
    if the binary is missing; callers map that to HTTP 500.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-i', str(input_path),
            '-ar', '16000', '-ac', '1', str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg conversion error: exit code {process.returncode}")
            return False

        return True
    except FileNotFoundError:
        logger.error("FFmpeg not installed in container! Run: apt-get install ffmpeg")
        return False

def _save_embedding_and_rebuild(tmp_save: str, avg_embeddings_numpy: np.ndarray, final_path: str):
    """Persist an averaged embedding atomically, then refresh the gallery.

    np.save appends '.npy' to tmp_save. The temp file must live on the same
    filesystem as final_path (the caller creates it inside SPEAKERS_DIR) so
    the final os.replace is an atomic same-filesystem rename — a cross-device
    move would degrade to copy+rename and briefly expose a half-written file
    to concurrent /identify requests. Runs in a worker thread via
    run_in_threadpool.
    """
    np.save(tmp_save, avg_embeddings_numpy)
    os.replace(tmp_save + ".npy", final_path)
    _rebuild_cache()

def _rebuild_cache():
    """Rescan SPEAKERS_DIR and restack all embeddings into the global matrix.

    Called at import time is NOT required — identify lazily triggers a rebuild
    when the matrix is still empty. Corrupted .npy files are skipped with a
    warning instead of failing enrollment/identification wholesale.
    """
    global _embedding_names, _embedding_matrix
    names = []
    tensors = []
    for speaker_file in sorted(SPEAKERS_DIR.glob("*.npy")):
        try:
            t = torch.tensor(np.load(speaker_file), device=device)
            t = F.normalize(t, p=2, dim=-1)
            names.append(speaker_file.stem)
            tensors.append(t)
        except Exception as e:
            logger.warning(f"Skipping corrupted {speaker_file.name}: {e}")
    with _cache_lock:
        _embedding_names = names
        _embedding_matrix = torch.stack(tensors) if tensors else None

@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    """Identify a speaker from a single uploaded audio file (no auth).

    Pipeline: stream to disk (enforcing the size cap) -> FFmpeg to 16 kHz mono
    WAV -> fbank features -> CAMPPlus embedding -> batched cosine similarity
    against every enrolled speaker -> best match above 0.4, else 'unknown'.
    Temp files are always removed in the finally block, even on errors.
    """
    # Never trust client-supplied paths: keep only the basename so crafted
    # names like '../../etc/passwd' cannot influence where bytes are written.
    safe_filename = os.path.basename(file.filename) if file.filename else "upload.raw"
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = os.path.splitext(safe_filename)[1] or ".raw"
    # Pre-create both spill files up front: raw upload + converted WAV.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp1, tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
        temp_input = tmp1.name
        temp_wav = tmp2.name
    
    try:
        # Stream the upload in 1 MB chunks instead of buffering it in RAM,
        # aborting with 413 as soon as the cumulative size exceeds the cap.
        file_size = 0
        async with await anyio.open_file(temp_input, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
                await buffer.write(chunk)

        if not await convert_to_wav(temp_input, temp_wav):
            raise HTTPException(status_code=500, detail="Failed to process audio format")

        signal, fs = await run_in_threadpool(torchaudio.load, temp_wav)
        # ~0.25 s minimum: shorter clips produce unstable embeddings.
        if signal.numel() == 0 or signal.shape[-1] < 4000:
            raise HTTPException(status_code=400, detail="Audio too short or empty")
        # Peak-normalize to 0.9: phone Opus recordings are much quieter than
        # desktop WAVs, and unnormalized level differences crush similarity.
        peak = signal.abs().max()
        if peak > 0:
            signal = signal / peak * 0.9
        fbank = compute_fbank(signal.to(device), fs)
        embedding = await run_in_threadpool(_model_inference, fbank)
        embedding = F.normalize(embedding, p=2, dim=-1)  # cosine == dot product now

        # Snapshot the gallery under the lock (matrix swap is atomic), then do
        # the heavy matmul outside it so enrollment is never blocked.
        with _cache_lock:
            names = _embedding_names
            matrix = _embedding_matrix
        
        if matrix is None:
            _rebuild_cache()
            with _cache_lock:
                names = _embedding_names
                matrix = _embedding_matrix
        
        if matrix is not None:
            scores = (embedding.squeeze(0) @ matrix.T).cpu().numpy()
            max_idx = scores.argmax()
            max_score = float(scores[max_idx])
            best_user = names[max_idx]
        else:
            max_score = 0.0
            best_user = "unknown"
        
        if max_score < 0.4:
            best_user = "unknown"  # below threshold we refuse to guess
            
        logger.info(f"Identified: {best_user} (Confidence: {max_score:.2f})")
        return IdentifyResponse(user_id=best_user, confidence=max_score)
        
    finally:
        # Disk cleanup off the event loop; runs even when exceptions propagate.
        if os.path.exists(temp_input):
            await run_in_threadpool(_safe_remove, temp_input)
        if os.path.exists(temp_wav):
            await run_in_threadpool(_safe_remove, temp_wav)

@app.get("/enroll", response_class=HTMLResponse)
async def enroll_form():
    """Serve the self-contained enrollment UI.

    The page is a single inline HTML string (CSS + vanilla JS, no build step,
    no static files) implementing an Apple-style design — DESIGN.md at the
    repo root is its spec. It records up to three samples via getUserMedia,
    encodes them to WAV client-side (encodeWAV) and POSTs them together with
    user_id and the API key to POST /enroll. Dark/light theme follows
    prefers-color-scheme and persists in localStorage.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Enrollment | Speaker ID</title>
<meta name="description" content="Enroll your voice profile for speaker identification">
<style>
:root {
  --c-canvas: #ffffff;
  --c-canvas-alt: #f5f5f7;
  --c-ink: #1d1d1f;
  --c-ink-secondary: #7a7a7a;
  --c-primary: #0066cc;
  --c-primary-rgb: 0, 102, 204;
  --c-primary-hover: #0071e3;
  --c-primary-on-dark: #2997ff;
  --c-on-primary: #ffffff;
  --c-hairline: #e0e0e0;
  --c-divider: #f0f0f0;
  --c-surface-pearl: #fafafc;
  --c-success: #34d34d;
  --c-success-rgb: 52, 211, 77;
  --c-error: #ff3b30;
  --c-error-rgb: 255, 59, 48;
  --c-warning: #ff9f0a;
  --c-warning-rgb: 255, 159, 10;
  --c-red-action: #ff3b30;
  --c-red-action-rgb: 255, 59, 48;
  --c-success-bg: #e8f8ee;
  --c-error-bg: #ffeeed;
  --t-display: 34px/1.47 -0.374px;
  --t-body: 17px/1.47 -0.374px;
  --t-caption: 14px/1.43 -0.224px;
  --t-caption-strong: 14px/1.29 -0.224px;
  --t-utility: 14px/1.29 -0.224px;
  --t-micro: 12px/1.0 -0.12px;
  --r-pill: 9999px;
  --r-lg: 18px;
  --r-sm: 8px;
  --r-md: 11px;
  --s-xs: 8px;
  --s-sm: 12px;
  --s-md: 17px;
  --s-lg: 24px;
  --s-xl: 32px;
  --s-xxl: 48px;
}
[data-theme="dark"] {
  --c-canvas: #272729;
  --c-canvas-alt: #2a2a2c;
  --c-ink: #ffffff;
  --c-ink-secondary: #98989d;
  --c-primary: #2997ff;
  --c-primary-rgb: 41, 151, 255;
  --c-primary-hover: #40a9ff;
  --c-on-primary: #ffffff;
  --c-hairline: #3a3a3c;
  --c-divider: #3a3a3c;
  --c-surface-pearl: #333336;
  --c-success: #30d158;
  --c-success-rgb: 48, 209, 88;
  --c-error: #ff453a;
  --c-error-rgb: 255, 69, 58;
  --c-warning: #ffd60a;
  --c-warning-rgb: 255, 214, 10;
  --c-red-action: #ff453a;
  --c-red-action-rgb: 255, 69, 58;
  --c-success-bg: #1a3a2a;
  --c-error-bg: #3a1a1a;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif;
  background: var(--c-canvas-alt);
  color: var(--c-ink);
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--s-lg);
  transition: background 0.3s, color 0.3s;
}

.theme-toggle {
  position: fixed;
  top: var(--s-lg);
  right: var(--s-lg);
  z-index: 100;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid var(--c-hairline);
  background: var(--c-canvas);
  color: var(--c-ink);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  transition: background 0.3s, border-color 0.3s;
}
.theme-toggle:hover {
  border-color: var(--c-primary);
}

.container {
  background: var(--c-canvas);
  border-radius: var(--r-lg);
  border: 1px solid var(--c-hairline);
  padding: var(--s-xxl) var(--s-xl);
  width: 100%;
  max-width: 560px;
  transition: background 0.3s, border-color 0.3s;
}

.header {
  text-align: center;
  margin-bottom: var(--s-xl);
}

.icon-wrap {
  width: 44px;
  height: 44px;
  margin: 0 auto var(--s-sm);
  background: var(--c-primary);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.icon-wrap svg {
  width: 22px;
  height: 22px;
  fill: var(--c-on-primary);
}

h1 {
  font: 600 var(--t-display);
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif;
  margin-bottom: var(--s-xs);
  letter-spacing: -0.374px;
}
.subtitle {
  font: 400 17px/1.47 -0.374px;
  color: var(--c-ink-secondary);
}

.progress {
  display: flex;
  gap: var(--s-xs);
  justify-content: center;
  margin-bottom: var(--s-lg);
}
.progress-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--c-hairline);
  transition: all 0.3s;
}
.progress-dot.active {
  background: var(--c-primary);
  transform: scale(1.3);
}
.progress-dot.completed {
  background: var(--c-primary);
  opacity: 0.6;
}

.form-group {
  margin-bottom: var(--s-lg);
}
label {
  display: block;
  font: 600 var(--t-caption);
  color: var(--c-ink);
  margin-bottom: var(--s-xs);
}
input[type="text"] {
  width: 100%;
  height: 44px;
  padding: 0 var(--s-md);
  border: 1px solid var(--c-hairline);
  border-radius: var(--r-pill);
  font: 400 var(--t-body);
  background: var(--c-canvas-alt);
  color: var(--c-ink);
  outline: none;
  transition: border-color 0.2s, background 0.3s;
}
input[type="text"]:focus {
  border-color: var(--c-primary);
  box-shadow: 0 0 0 3px rgba(var(--c-primary-rgb), 0.15);
}
input[type="text"]::placeholder {
  color: var(--c-ink-secondary);
  opacity: 0.6;
}

.samples-container {
  display: flex;
  flex-direction: column;
  gap: var(--s-sm);
  margin-bottom: var(--s-sm);
}

.sample {
  background: var(--c-canvas-alt);
  border: 1px solid var(--c-divider);
  border-radius: var(--r-lg);
  padding: var(--s-lg);
  transition: border-color 0.2s, background 0.3s;
}
.sample:hover {
  border-color: var(--c-hairline);
}

.sample-header {
  display: flex;
  align-items: center;
  gap: var(--s-xs);
  margin-bottom: var(--s-sm);
}
.sample-num {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--c-primary);
  color: var(--c-on-primary);
  font: 600 var(--t-micro);
  display: flex;
  align-items: center;
  justify-content: center;
}
.sample h3 {
  font: 600 var(--t-caption-strong);
  color: var(--c-ink);
}

.text-box {
  background: var(--c-canvas);
  border-radius: var(--r-sm);
  padding: var(--s-sm) var(--s-md);
  margin-bottom: var(--s-sm);
  border-left: 3px solid var(--c-primary);
}
.text-box p {
  font: 400 14px/1.5 -0.224px;
  color: var(--c-ink-secondary);
}

.sample-actions {
  display: flex;
  gap: var(--s-xs);
  flex-wrap: wrap;
  align-items: center;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: none;
  cursor: pointer;
  font: 400 var(--t-utility);
  transition: all 0.2s;
  white-space: nowrap;
}

.btn-record {
  background: var(--c-primary);
  color: var(--c-on-primary);
  padding: 6px 14px;
  border-radius: var(--r-pill);
}
.btn-record:hover:not(:disabled) {
  background: var(--c-primary-hover);
}
.btn-record:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.btn-stop {
  background: var(--c-red-action);
  color: white;
  padding: 6px 14px;
  border-radius: var(--r-pill);
}
.btn-stop:hover:not(:disabled) {
  opacity: 0.85;
}
.btn-stop:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn-remove {
  background: transparent;
  color: var(--c-ink-secondary);
  padding: 6px 10px;
  border-radius: var(--r-pill);
  font-size: 12px;
  margin-left: auto;
}
.btn-remove:hover {
  color: var(--c-error);
}

.recording-indicator {
  display: none;
  align-items: center;
  gap: var(--s-xs);
  margin-top: var(--s-xs);
  padding: var(--s-xs) var(--s-sm);
  background: rgba(var(--c-warning-rgb), 0.12);
  border-radius: var(--r-pill);
  width: fit-content;
}
.recording-indicator.active {
  display: flex;
}
.recording-indicator .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--c-warning);
  animation: blink 1s infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.recording-indicator .timer {
  font: 700 16px/1;
  color: var(--c-warning);
  font-variant-numeric: tabular-nums;
}
.recording-indicator .label {
  font: 400 var(--t-micro);
  color: var(--c-warning);
}

audio {
  width: 100%;
  margin-top: var(--s-xs);
  border-radius: var(--r-sm);
  height: 36px;
}
audio::-webkit-media-controls-panel {
  background: var(--c-canvas-alt);
}

.btn-add {
  width: 100%;
  padding: 10px;
  border-radius: var(--r-pill);
  border: 1px solid var(--c-primary);
  background: transparent;
  color: var(--c-primary);
  font: 400 var(--t-body);
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
}
.btn-add:hover:not(:disabled) {
  background: rgba(var(--c-primary-rgb), 0.08);
}
.btn-add:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn-submit {
  width: 100%;
  padding: 12px 22px;
  border-radius: var(--r-pill);
  border: none;
  background: var(--c-primary);
  color: var(--c-on-primary);
  font: 400 var(--t-body);
  cursor: pointer;
  margin-top: var(--s-lg);
  transition: background 0.2s, opacity 0.2s;
}
.btn-submit:hover:not(:disabled) {
  background: var(--c-primary-hover);
}
.btn-submit:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.status-message {
  margin-top: var(--s-lg);
  padding: var(--s-sm) var(--s-md);
  border-radius: var(--r-md);
  display: none;
  font: 400 15px/1.5;
  text-align: center;
}
.status-message.success {
  display: block;
  background: var(--c-success-bg);
  color: var(--c-success);
  border: 1px solid rgba(var(--c-success-rgb), 0.3);
}
.status-message.error {
  display: block;
  background: var(--c-error-bg);
  color: var(--c-error);
  border: 1px solid rgba(var(--c-error-rgb), 0.3);
}

@media (max-width: 640px) {
  body { padding: var(--s-sm); }
  .container { padding: var(--s-lg) var(--s-md); }
  h1 { font-size: 28px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
</head>
<body>
<button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme">🌙</button>

<div class="container">
  <div class="header">
    <div class="icon-wrap">
      <svg viewBox="0 0 24 24">
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
      </svg>
    </div>
    <h1>Voice Enrollment</h1>
    <p class="subtitle">Record 3 speech samples to create your voice profile</p>
  </div>

  <div class="progress" id="progress-container">
    <div class="progress-dot" data-step="1"></div>
    <div class="progress-dot" data-step="2"></div>
    <div class="progress-dot" data-step="3"></div>
  </div>

  <div class="form-group">
    <label for="user_id">Username</label>
    <input type="text" id="user_id" placeholder="e.g. alexander" value="">
  </div>

  <div class="form-group">
    <label for="api_key">API Key</label>
    <input type="password" id="api_key" placeholder="Enter API key" value="" style="width: 100%; height: 44px; padding: 0 var(--s-md); border: 1px solid var(--c-hairline); border-radius: var(--r-pill); font: 400 var(--t-body); background: var(--c-canvas-alt); color: var(--c-ink); outline: none; transition: border-color 0.2s, background 0.3s;">
  </div>

  <div class="samples-container" id="samples-container"></div>

  <button class="btn-add" id="add-sample-btn" onclick="addSample()">+ Add sample</button>

  <button class="btn-submit" id="submit-btn" onclick="submitEnroll()" disabled>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle">
      <path d="M20 6L9 17l-5-5"/>
    </svg>
    Register Voice
  </button>

  <div id="status" class="status-message"></div>
</div>

<script>
// Enrollment UI logic: records PCM via a ScriptProcessorNode at the browser's
// native sample rate, converts Int16 frames into a WAV blob (encodeWAV), and
// uploads everything as multipart/form-data with an X-API-Key header.
const TEXTS = [
  "Hello, computer! I am setting up my voice profile for the smart home system. This audio sample will help the neural network remember my voice.",
  "The weather is great today, the sun is shining and the birds are singing outside. I hope the system recognizes my voice without errors even in a noisy room.",
  "One, two, three, four, five, six, seven, eight, nine, ten. I am speaking with different intonation to make the sample as complete and high-quality as possible."
];

const themeToggle = document.getElementById('theme-toggle');

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
  localStorage.setItem('theme', theme);
}

function toggleTheme() {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  setTheme(next);
}

(function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved) { setTheme(saved); return; }
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) setTheme('dark');
  else setTheme('light');
})();

let samples = [];
let audioChunks = [];
let recordingTimer;
let recordingDuration = 0;
let recordingStartTime = 0;
let recordingContext;
let recordingSource;
let recordingScriptProcessor;
let recording = false;
let recordingSampleRate = 16000;
let mediaStream = null;

function addSample() {
  const index = samples.length;
  if (index >= 3) return;

  const d = document.createElement('div');
  d.className = 'sample';
  d.innerHTML = `
    <div class="sample-header">
      <span class="sample-num">${index + 1}</span>
      <h3>Sample ${index + 1} of 3</h3>
      <button class="btn btn-remove remove-btn" onclick="removeSample(${index})" style="display:none">Remove</button>
    </div>
    <div class="text-box"><p>${TEXTS[index]}</p></div>
    <div class="sample-actions">
      <button class="btn btn-record" onclick="startRecording(${index})">● Record</button>
      <button class="btn btn-stop" onclick="stopRecording(${index})" disabled>■ Stop</button>
    </div>
    <div class="recording-indicator" id="recording-${index}">
      <span class="dot"></span>
      <span class="timer" id="timer-${index}">00:00</span>
      <span class="label">Recording…</span>
    </div>
    <audio id="audio-${index}" controls style="display:none"></audio>
  `;
  document.getElementById('samples-container').appendChild(d);
  samples.push({ index, recorded: false, blob: null });
}

async function startRecording(index) {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showStatus('Error: microphone not available in this browser', 'error');
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    mediaStream = stream;
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    if (ctx.state === 'suspended') await ctx.resume();

    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(ctx.destination);

    audioChunks = [];
    recordingSampleRate = ctx.sampleRate;
    recordingContext = ctx;
    recordingSource = source;
    recordingScriptProcessor = processor;

    processor.onaudioprocess = (e) => {
      if (!recording) return;
      const left = e.inputBuffer.getChannelData(0);
      const buf = new Int16Array(left.length);
      for (let i = 0; i < left.length; i++) {
        const s = Math.max(-1, Math.min(1, left[i]));
        buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      audioChunks.push(buf);
    };

    recording = true;
    recordingStartTime = Date.now();
    const timerEl = document.getElementById('timer-' + index);

    const btns = document.querySelectorAll('#samples-container .sample')[index].querySelectorAll('.sample-actions button');
    btns[0].disabled = true;
    btns[1].disabled = false;

    document.getElementById('recording-' + index).classList.add('active');

    recordingTimer = setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
      const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const s = String(elapsed % 60).padStart(2, '0');
      timerEl.textContent = m + ':' + s;
    }, 200);
  } catch (err) {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    showStatus('Microphone access error: ' + err.message, 'error');
  }
}

function stopRecording(index) {
  recording = false;
  if (recordingScriptProcessor) { recordingScriptProcessor.disconnect(); recordingScriptProcessor = null; }
  if (recordingSource) recordingSource.disconnect();
  if (recordingContext) { recordingContext.close(); recordingContext = null; }
  clearInterval(recordingTimer);
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

  try {
    if (audioChunks.length > 0) {
      const sr = recordingSampleRate || 16000;
      const blob = encodeWAV(sr, audioChunks);
      const audio = document.getElementById('audio-' + index);
      audio.src = URL.createObjectURL(blob);
      audio.style.display = 'block';

      const sample = document.querySelectorAll('#samples-container .sample')[index];
      sample.querySelector('.remove-btn').style.display = '';

      samples[index].recorded = true;
      samples[index].blob = blob;
      updateSubmitButton();
    } else {
      alert('No audio data');
    }
  } catch (err) {
    alert('Save error: ' + err.message);
  }

  const btns = document.querySelectorAll('#samples-container .sample')[index].querySelectorAll('.sample-actions button');
  btns[0].disabled = false;
  btns[1].disabled = true;
  document.getElementById('recording-' + index).classList.remove('active');
}

function removeSample(index) {
  const el = document.querySelectorAll('#samples-container .sample')[index];
  if (!el) return;
  el.remove();
  samples.splice(index, 1);
  updateSampleNumbers();
  updateSubmitButton();
}

function updateSampleNumbers() {
  const divs = document.querySelectorAll('#samples-container .sample');
  divs.forEach((d, i) => {
    d.querySelector('h3').textContent = 'Sample ' + (i + 1) + ' of 3';
    d.querySelector('.sample-num').textContent = i + 1;
    d.querySelector('.text-box p').textContent = TEXTS[i];
  });
}

function updateSubmitButton() {
  document.getElementById('submit-btn').disabled = samples.length < 1 || !samples.every(s => s.recorded);
}

function showStatus(msg, type) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status-message ' + type;
}

async function submitEnroll() {
  const uid = document.getElementById('user_id').value.trim();
  const apiKey = document.getElementById('api_key').value.trim();
  if (!uid) { showStatus('Enter a username', 'error'); return; }
  if (samples.length < 1) { showStatus('Record at least one sample', 'error'); return; }
  if (!apiKey) { showStatus('Enter an API key', 'error'); return; }

  const fd = new FormData();
  fd.append('user_id', uid);
  samples.forEach((s, i) => {
    fd.append('files', new Blob([s.blob], { type: 'audio/wav' }), 'sample_' + (i + 1) + '.wav');
  });

  try {
    const r = await fetch('/enroll', {
      method: 'POST',
      headers: {
        'X-API-Key': apiKey
      },
      body: fd
    });
    const data = await r.json();
    if (r.ok) {
      showStatus('✓ Success! ' + data.status, 'success');
      document.getElementById('samples-container').innerHTML = '';
      samples = [];
      updateSubmitButton();
    } else {
      showStatus('✗ Error: ' + (data.detail || 'Unknown error'), 'error');
    }
  } catch (err) {
    showStatus('Connection error: ' + err.message, 'error');
  }
}

function encodeWAV(sr, chunks) {
  const ch = 1, bps = 2, ba = ch * bps;
  let size = 0;
  chunks.forEach(c => size += c.byteLength);
  const buf = new ArrayBuffer(44 + size);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); v.setUint32(4, 36 + size, true);
  ws(8, 'WAVE'); ws(12, 'fmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, ch, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * ba, true);
  v.setUint16(32, ba, true); v.setUint16(34, bps * 8, true);
  ws(36, 'data'); v.setUint32(40, size, true);
  let off = 44;
  chunks.forEach(c => { const d = new Int16Array(c); for (let i = 0; i < d.length; i++) { v.setInt16(off, d[i], true); off += 2; } });
  return new Blob([buf], { type: 'audio/wav' });
}

addSample();
</script>
</body>
</html>"""
    return html


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), files: list[UploadFile] = File(...), api_key: str = Security(get_api_key)):
    """Enroll/replace a speaker from one or more audio samples (API-key auth).

    Every file goes through the same pipeline as /identify; the resulting
    embeddings are averaged and L2-normalized into ONE voice profile stored as
    SPEAKERS_DIR/<user_id>.npy. Re-enrolling an existing user_id atomically
    overwrites their profile. Any per-file failure aborts the whole enrollment;
    the finally block removes all temp files created so far.
    """
    # Sanitize user_id: basename first, then whitelist regex -> the value can
    # never escape SPEAKERS_DIR or smuggle unexpected characters into filenames.
    user_id = os.path.basename(user_id)
    if not user_id or not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file is required")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail="Too many files uploaded")
    
    embeddings_list = []
    temp_files = []
    
    try:
        for file in files:
            safe_filename = os.path.basename(file.filename) if file.filename else "upload.raw"
            if not safe_filename or safe_filename in (".", ".."):
                raise HTTPException(status_code=400, detail="Invalid filename")

            ext = os.path.splitext(safe_filename)[1] or ".raw"
            # Pair of spill files per upload: raw bytes + converted WAV.
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp1, tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
                temp_input = tmp1.name
                temp_wav = tmp2.name
            temp_files.extend([temp_input, temp_wav])
            
            file_size = 0
            # Same chunked streaming + size cap as /identify.
            async with await anyio.open_file(temp_input, "wb") as buffer:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > MAX_FILE_SIZE:
                        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
                    await buffer.write(chunk)

            if not await convert_to_wav(temp_input, temp_wav):
                raise HTTPException(status_code=500, detail="Failed to process audio format")

            signal, fs = await run_in_threadpool(torchaudio.load, temp_wav)
            if signal.numel() == 0 or signal.shape[-1] < 4000:
                raise HTTPException(status_code=400, detail="Audio too short or empty")
            # Peak normalization identical to /identify so enroll and identify
            # embeddings live in the same feature distribution.
            peak = signal.abs().max()
            if peak > 0:
                signal = signal / peak * 0.9
            fbank = compute_fbank(signal.to(device), fs)
            embedding = await run_in_threadpool(_model_inference, fbank)
            embedding = F.normalize(embedding, p=2, dim=-1)
            embeddings_list.append(embedding.squeeze().cpu())

        # Average all samples into one centroid embedding, re-normalize so it
        # stays a unit vector on the hypersphere.
        avg_embeddings = torch.stack(embeddings_list).mean(dim=0)
        avg_embeddings = F.normalize(avg_embeddings, p=2, dim=-1)
        # Atomic publish: temp file is created INSIDE SPEAKERS_DIR so the final
        # os.replace in _save_embedding_and_rebuild is a same-filesystem rename.
        # (A /tmp temp file would cross filesystems and lose atomicity.) The
        # dot-prefix keeps partial files out of the way if we crash mid-write;
        # _rebuild_cache skips unreadable leftovers with a warning.
        tmp_save = str(SPEAKERS_DIR / f".tmp-{uuid.uuid4()}")
        final_path = str(SPEAKERS_DIR / f"{user_id}.npy")
        await run_in_threadpool(_save_embedding_and_rebuild, tmp_save, avg_embeddings.numpy(), final_path)

        logger.info(f"Voice enrolled: {user_id} ({len(files)} samples)")
        return EnrollResponse(status="success", user_id=user_id)
        
    finally:
        # Remove every spill file created during this request (thread pool so
        # unlink syscalls never stall the event loop).
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                await run_in_threadpool(_safe_remove, temp_file)

@app.get("/health")
async def health():
    """Readiness probe for the Docker healthcheck: 200 once the model has been
    loaded and warmed up, 503 before that."""
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ok"}

if __name__ == "__main__":
    # TLS is opt-in: mount certs via env vars (compose can inject them). With
    # no certs we still serve plain HTTP but warn loudly — production setups
    # should either set SSL_* or terminate TLS at a reverse proxy.
    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    if ssl_keyfile and ssl_certfile:
        logger.info("Starting server with SSL/TLS enabled.")
        uvicorn.run(app, host="0.0.0.0", port=8001, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        logger.warning("WARNING: Starting server without SSL/TLS. In production, ensure this service is behind a reverse proxy (like Nginx) that handles HTTPS.")
        uvicorn.run(app, host="0.0.0.0", port=8001)
