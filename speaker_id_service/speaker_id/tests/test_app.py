import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Also we need to make sure the app can be imported if it assumes running from /app
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Create mocked version of the app to avoid downloading the model
with patch('urllib.request.urlretrieve'), \
     patch('torch.load', return_value={}), \
     patch('campplus_model.CAMPPlus.load_state_dict'):
    from app import app, IdentifyResponse, EnrollResponse

client = TestClient(app)

@pytest.fixture
def mock_dependencies(tmp_path):
    # Patch SPEAKERS_DIR to use a temporary directory
    mock_speakers_dir = tmp_path / "speakers"
    mock_speakers_dir.mkdir()

    with patch('app.SPEAKERS_DIR', mock_speakers_dir), \
         patch('app.convert_to_wav', return_value=True), \
         patch('app.torchaudio.load') as mock_load, \
         patch('app.model') as mock_model:

        # Setup mock audio
        dummy_signal = torch.ones(1, 16000)
        mock_load.return_value = (dummy_signal, 16000)

        # Setup mock model output
        # Model should return a 512-dim embedding
        dummy_embedding = torch.ones(1, 512)
        mock_model.return_value = dummy_embedding

        yield {
            "speakers_dir": mock_speakers_dir,
            "mock_load": mock_load,
            "mock_model": mock_model
        }

def test_identify_no_file():
    response = client.post("/identify")
    assert response.status_code == 422 # Unprocessable Entity (Missing field)

def test_enroll_no_files():
    response = client.post("/enroll", data={"user_id": "test_user"})
    assert response.status_code == 422 # Unprocessable Entity

def test_enroll_success(mock_dependencies):
    speakers_dir = mock_dependencies["speakers_dir"]

    # Send a mock file
    files = [("files", ("test.wav", b"dummy audio content", "audio/wav"))]
    data = {"user_id": "test_user"}

    response = client.post("/enroll", data=data, files=files)

    assert response.status_code == 200
    assert response.json() == {"status": "success", "user_id": "test_user"}

    # Check if a .npy file was created
    npy_file = speakers_dir / "test_user.npy"
    assert npy_file.exists()

    # The saved embedding should be normalized
    saved_emb = np.load(npy_file)
    assert saved_emb.shape == (512,)

def test_identify_success(mock_dependencies):
    speakers_dir = mock_dependencies["speakers_dir"]

    # First enroll a user manually
    # The model output is a tensor of ones, normalized
    dummy_embedding = torch.ones(1, 512)
    normalized_emb = torch.nn.functional.normalize(dummy_embedding, p=2, dim=-1)
    np.save(speakers_dir / "known_user.npy", normalized_emb.squeeze().numpy())

    # Now try to identify
    # We use a mocked model that will output the exact same vector, so cosine sim = 1.0 > 0.4
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    response = client.post("/identify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "known_user"
    assert data["confidence"] > 0.99 # Should be exactly 1.0, accounting for float inaccuracy

def test_identify_unknown(mock_dependencies):
    speakers_dir = mock_dependencies["speakers_dir"]

    # Enroll a user with a COMPLETELY DIFFERENT embedding
    different_embedding = -torch.ones(1, 512)
    normalized_emb = torch.nn.functional.normalize(different_embedding, p=2, dim=-1)
    np.save(speakers_dir / "other_user.npy", normalized_emb.squeeze().numpy())

    # Model will output ones, enrolled is minus ones. Cosine sim = -1.0 < 0.4
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    response = client.post("/identify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "unknown"
    assert data["confidence"] < 0.4

def test_identify_audio_too_short(mock_dependencies):
    # Mock torchaudio.load to return a short signal
    mock_dependencies["mock_load"].return_value = (torch.ones(1, 100), 16000)

    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    response = client.post("/identify", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio too short or empty"

def test_enroll_audio_too_short(mock_dependencies):
    # Mock torchaudio.load to return a short signal
    mock_dependencies["mock_load"].return_value = (torch.ones(1, 100), 16000)

    files = [("files", ("test.wav", b"dummy audio content", "audio/wav"))]
    data = {"user_id": "test_user"}
    response = client.post("/enroll", data=data, files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio too short or empty"

def test_enroll_form_get():
    response = client.get("/enroll")
    assert response.status_code == 200
    assert "Голосовая регистрация" in response.text
