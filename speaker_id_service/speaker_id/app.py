import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import uuid
import sys
sys.path.insert(0, "/app")
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from campplus_model import CAMPPlus
from pathlib import Path
import logging
import shutil
import uvicorn
import subprocess

# Setup and logging
logging.basicConfig(level=logging.INFO)
for _logger in ["httpx", "urllib3", "filelock"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI()

SPEAKERS_DIR = Path("/app/speakers")
MODELS_DIR = Path("/app/models/speaker_id")
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# CUDA check for GPU (P104-100)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"--- Speaker ID Service ---")
logger.info(f"Device: {device}")
if device != "cpu":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Load CAM++ model (WeSpeaker, VoxCeleb)
logger.info("Loading CAMPPlus model...")
model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
ckpt_path = MODELS_DIR / "campplus_avg_model.pt"
if not ckpt_path.exists():
    import urllib.request
    url = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/avg_model.pt"
    logger.info(f"Downloading CAM++ from {url}")
    urllib.request.urlretrieve(url, str(ckpt_path))
ckpt = torch.load(str(ckpt_path), map_location=device)
state_dict = {}
for k, v in ckpt.items():
    k = k.replace("module.", "")
    if not k.startswith("projection"):
        state_dict[k] = v
model.load_state_dict(state_dict)
model.to(device)
model.eval()
logger.info("CAMPPlus model successfully loaded!")

def compute_fbank(signal: torch.Tensor, fs: int) -> torch.Tensor:
    """Convert raw audio to 80-dim fbank for CAM++."""
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000).to(signal.device)
        signal = resampler(signal)
        fs = 16000
    fbank = kaldi.fbank(signal, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=fs)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    return fbank.unsqueeze(0)  # (1, num_frames, feat_dim)

class IdentifyResponse(BaseModel):
    user_id: str
    confidence: float

class EnrollResponse(BaseModel):
    status: str
    user_id: str

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio to 16000Hz Mono WAV via FFmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        logger.error("FFmpeg not installed in container! Run: apt-get install ffmpeg")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion error: {e}")
        return False

@app.post("/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...)):
    """Identify speaker from audio file."""
    # Use UUID to prevent file collisions under concurrent requests
    req_id = str(uuid.uuid4())
    temp_input = f"/tmp/{req_id}_{file.filename}"
    temp_wav = f"/tmp/{req_id}_processed.wav"
    
    # Save incoming file (raw opus)
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Convert to WAV (16kHz mono)
        if not convert_to_wav(temp_input, temp_wav):
            raise HTTPException(status_code=500, detail="Failed to process audio format")

        signal, fs = torchaudio.load(temp_wav)
        if signal.numel() == 0 or signal.shape[-1] < 4000:
            raise HTTPException(status_code=400, detail="Audio too short or empty")
        # Peak normalization
        peak = signal.abs().max()
        if peak > 0:
            signal = signal / peak * 0.9
        # Fbank + CAM++
        fbank = compute_fbank(signal.to(device), fs)
        with torch.no_grad():
            embedding = model(fbank)
        embedding = F.normalize(embedding, p=2, dim=-1)

        max_score = 0.0
        best_user = "unknown"
        
        for speaker_file in SPEAKERS_DIR.glob("*.npy"):
            enrolled = torch.tensor(np.load(speaker_file)).to(device)
            enrolled = F.normalize(enrolled, p=2, dim=-1)
            score = F.cosine_similarity(embedding.squeeze(), enrolled.squeeze(), dim=0).item()
            if score > max_score:
                max_score = score
                best_user = speaker_file.stem
        
        # Confidence threshold check
        if max_score < 0.4:
            best_user = "unknown"
            
        logger.info(f"Identified: {best_user} (Confidence: {max_score:.2f})")
        return IdentifyResponse(user_id=best_user, confidence=max_score)
        
    finally:
        # Cleanup temporary files reliably
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

@app.get("/enroll", response_class=HTMLResponse)
async def enroll_form():