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
import time
import collections
import asyncio
import secrets
import tempfile
import anyio
import hashlib
sys.path.insert(0, "/app")
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security, Request, Depends
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, FileResponse
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
    except OSError as e:
        logger.warning(f"Failed to remove {path}: {e}")

for _logger in ["httpx", "urllib3", "filelock"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(_setup_model_and_warmup)
    yield

app = FastAPI(lifespan=lifespan)

allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "http://localhost,http://localhost:8001")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# Define model configuration globally
model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
_model_ready = False

def _setup_model_and_warmup():
    global _model_ready
    logger.info("Loading CAMPPlus model...")
    ckpt_path = MODELS_DIR / "campplus_avg_model.pt"
    if not ckpt_path.exists():
        # First start: fetch the official VoxCeleb checkpoint (63 MB) from HF Hub.
        import urllib.request
        import urllib.parse
        url = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/avg_model.pt"
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError(f"Refusing to download from untrusted scheme: {url}")
        logger.info(f"Downloading CAM++ from {url}")
        urllib.request.urlretrieve(url, str(ckpt_path))  # nosec B310

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
# Async lock against cache stampedes: concurrent /identify requests hitting an
# empty gallery rebuild the matrix only once (double-checked locking below).
_async_cache_lock = asyncio.Lock()
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
    if not expected_api_key or not api_key or not secrets.compare_digest(api_key, expected_api_key):
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

    Note on dither=0.0: historically dither=1.0 was used following the classic
    Kaldi convention where waveforms are int16-scaled. Our waveform is a
    peak-normalized float in [-1, 1], so torchaudio's default added Gaussian
    noise of std ~1.0 directly onto it, causing slightly non-deterministic
    embeddings. Setting dither=0.0 fixes this for float inputs.
    """
    if fs != 16000:
        global _resampler_16k
        if _resampler_16k.orig_freq != fs or _resampler_16k.new_freq != 16000:
            _resampler_16k = torchaudio.transforms.Resample(fs, 16000).to(signal.device)
        signal = _resampler_16k(signal)
        fs = 16000
    fbank = kaldi.fbank(signal, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=fs)
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

def _save_embedding_and_rebuild(tmp_save: str, avg_embeddings_numpy: np.ndarray, final_path: str, user_id: str):
    """Persist an averaged embedding atomically, then incrementally update the gallery.

    np.save appends '.npy' to tmp_save. The temp file must live on the same
    filesystem as final_path (the caller creates it inside SPEAKERS_DIR) so
    the final os.replace is an atomic same-filesystem rename — a cross-device
    move would degrade to copy+rename and briefly expose a half-written file
    to concurrent /identify requests. Runs in a worker thread via
    run_in_threadpool.
    """
    np.save(tmp_save, avg_embeddings_numpy)
    os.replace(tmp_save + ".npy", final_path)

    global _embedding_names, _embedding_matrix

    # Convert exactly like _rebuild_cache does
    t = torch.tensor(avg_embeddings_numpy, device=device)
    t = F.normalize(t, p=2, dim=-1)

    with _cache_lock:
        if user_id in _embedding_names:
            idx = _embedding_names.index(user_id)
            if _embedding_matrix is not None:
                # Modifying a cloned tensor ensures readers concurrent to this block
                # still see a consistent (old) matrix.
                new_matrix = _embedding_matrix.clone()
                new_matrix[idx] = t
                _embedding_matrix = new_matrix
        else:
            _embedding_names = _embedding_names + [user_id]
            if _embedding_matrix is None:
                _embedding_matrix = t.unsqueeze(0)
            else:
                _embedding_matrix = torch.cat([_embedding_matrix, t.unsqueeze(0)], dim=0)

def _load_all_embeddings_sync():
    """Synchronous core for loading all embeddings. Returns names, matrix."""
    names = []
    tensors = []
    for speaker_file in sorted(SPEAKERS_DIR.glob("*.npy")):
        try:
            t = torch.tensor(np.load(speaker_file, allow_pickle=False), device=device)
            t = F.normalize(t, p=2, dim=-1)
            names.append(speaker_file.stem)
            tensors.append(t)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Skipping corrupted {speaker_file.name}: {e}")
    matrix = torch.stack(tensors) if tensors else None
    return names, matrix

async def _rebuild_cache():
    """Rescan SPEAKERS_DIR and restack all embeddings into the global matrix.

    Called at import time is NOT required — identify lazily triggers a rebuild
    when the matrix is still empty. Corrupted .npy files are skipped with a
    warning instead of failing enrollment/identification wholesale.
    """
    global _embedding_names, _embedding_matrix
    names, matrix = await run_in_threadpool(_load_all_embeddings_sync)
    with _cache_lock:
        _embedding_names = names
        _embedding_matrix = matrix


# ---------------------------------------------------------------------------
# Rate Limiting
# Simple in-memory rate limiting per IP address.
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 10
_rate_limits = collections.defaultdict(list)
_rate_limit_lock = threading.Lock()

async def check_rate_limit(request: Request):
    # Support reverse proxies
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")

    now = time.time()
    with _rate_limit_lock:
        # Clean up old entries and remove empty IPs to prevent memory leak
        for ip in list(_rate_limits.keys()):
            _rate_limits[ip] = [ts for ts in _rate_limits[ip] if now - ts < RATE_LIMIT_WINDOW]
            if not _rate_limits[ip]:
                del _rate_limits[ip]

        if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(status_code=429, detail="Too Many Requests")

        _rate_limits[client_ip].append(now)

async def process_audio_file(file: UploadFile) -> tuple[torch.Tensor, list[str]]:
    """Helper to process an uploaded audio file into an embedding.
    Returns the normalized embedding and a list of temporary file paths created."""
    safe_filename = os.path.basename(file.filename) if file.filename else "upload.raw"
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = os.path.splitext(safe_filename)[1] or ".raw"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp1, tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
        temp_input = tmp1.name
        temp_wav = tmp2.name
    
    temp_files = [temp_input, temp_wav]

    try:
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
            raise HTTPException(status_code=500, detail="Audio conversion failed")

        signal, fs = await run_in_threadpool(torchaudio.load, temp_wav)
        if signal.numel() == 0 or signal.shape[-1] < 4000:
            raise HTTPException(status_code=400, detail="Audio too short or empty")

        peak = signal.abs().max()
        if peak > 0:
            signal = signal / peak * 0.9
        fbank = compute_fbank(signal.to(device), fs)
        embedding = await run_in_threadpool(_model_inference, fbank)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding, temp_files
    except Exception:
        # On error during processing, clean up temp files immediately
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                await run_in_threadpool(_safe_remove, temp_file)
        raise


@app.post("/identify", response_model=IdentifyResponse, dependencies=[Depends(check_rate_limit)])
async def identify(file: UploadFile = File(...)):
    """Identify a speaker from a single uploaded audio file (no auth).

    Pipeline: stream to disk (enforcing the size cap) -> FFmpeg to 16 kHz mono
    WAV -> fbank features -> CAMPPlus embedding -> batched cosine similarity
    against every enrolled speaker -> best match above 0.4, else 'unknown'.
    Temp files are always removed in the finally block, even on errors.
    """
    temp_files = []
    try:
        embedding, temp_files = await process_audio_file(file)

        # Snapshot the gallery under the lock (matrix swap is atomic), then do
        # the heavy matmul outside it so enrollment is never blocked.
        with _cache_lock:
            names = _embedding_names
            matrix = _embedding_matrix
        
        if matrix is None:
            async with _async_cache_lock:
                # Double-check inside lock in case another request already rebuilt it
                with _cache_lock:
                    if _embedding_matrix is not None:
                        matrix = _embedding_matrix
                        names = _embedding_names

                if matrix is None:
                    await _rebuild_cache()
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
        # File removal happens on a thread pool so we don't stall the async
        # event loop with slow unlink syscalls.
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                await run_in_threadpool(_safe_remove, temp_file)

@app.get("/enroll", response_class=FileResponse)
async def enroll_form():
    """Serve the self-contained enrollment UI.

    The page is a static HTML file implementing an Apple-style design — DESIGN.md at the
    repo root is its spec. It records up to three samples via getUserMedia,
    encodes them to WAV client-side (encodeWAV) and POSTs them together with
    user_id and the API key to POST /enroll. Dark/light theme follows
    prefers-color-scheme and persists in localStorage.
    """
    html_path = Path(__file__).parent / "enroll.html"
    return FileResponse(html_path)


@app.post("/enroll", response_model=EnrollResponse, dependencies=[Depends(check_rate_limit)])
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
            embedding, t_files = await process_audio_file(file)
            temp_files.extend(t_files)
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
        await run_in_threadpool(_save_embedding_and_rebuild, tmp_save, avg_embeddings.numpy(), final_path, user_id)

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
