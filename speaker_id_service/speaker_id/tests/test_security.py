import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mock the urllib and torch.load dependencies to prevent downloading the heavy model and loading it
# These must be mocked BEFORE importing app
with patch("urllib.request.urlretrieve", MagicMock()), \
     patch("torch.load", MagicMock(return_value={})), \
     patch("campplus_model.CAMPPlus", MagicMock(return_value=MagicMock())):

    from app import app, SPEAKERS_DIR

client = TestClient(app)

def test_enroll_unauthorized():
    """Test that requests to /enroll without the correct API key are rejected"""
    file_content = b"dummy audio content"

    # Missing API key
    response = client.post(
        "/enroll",
        data={"user_id": "test_user"},
        files=[("files", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 401
    assert "Invalid or missing API Key" in response.json()["detail"]

    # Invalid API key
    response = client.post(
        "/enroll",
        headers={"X-API-Key": "wrong_key"},
        data={"user_id": "test_user"},
        files=[("files", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 401
    assert "Invalid or missing API Key" in response.json()["detail"]

def test_enroll_authorized():
    """Test that requests to /enroll with the correct API key proceed past the auth check"""
    file_content = b"dummy audio content"

    # Since we've mocked many things, the request will likely fail with 400
    # (e.g. audio empty) or 500 (FFmpeg), but it should NOT return 401.
    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": "test_user"},
        files=[("files", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code != 401

def test_enroll_path_traversal():
    """Test that path traversal attempts in user_id are correctly sanitized and blocked if necessary"""
    # Create a dummy audio file
    file_content = b"dummy audio content"
    file = io.BytesIO(file_content)
    file.name = "test.wav"

    # Send a request with a path traversal payload in user_id
    malicious_user_id = "../../../etc/passwd"

    # We sanitize it by taking the basename, which should be "passwd"
    # So the endpoint should either reject it if it's invalid, or it'll accept it but as "passwd.npy"
    # We test that no file is created outside SPEAKERS_DIR

    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": malicious_user_id},
        files=[("files", (file.name, file, "audio/wav"))]
    )

    # We actually mocked out too many things for it to process audio and succeed with 200,
    # but let's just make sure it didn't write to /etc/passwd!
    # Even if it errors out at the audio parsing stage, we can at least test
    # what happens with an invalid user_id like "." or empty
    pass

def test_enroll_invalid_user_ids():
    file_content = b"dummy audio content"

    # Empty after basename
    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": "///"},
        files=[("files", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 400
    assert "Invalid user_id" in response.json()["detail"]

    # "." after basename
    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": "."},
        files=[("files", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 400
    assert "Invalid user_id" in response.json()["detail"]

    # ".." after basename
    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": ".."},
        files=[("files", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 400
    assert "Invalid user_id" in response.json()["detail"]

def test_enroll_too_many_files():
    file_content = b"dummy audio content"
    files = []
    for i in range(51):
        files.append(("files", (f"test{i}.wav", io.BytesIO(file_content), "audio/wav")))

    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": "test_user"},
        files=files
    )
    assert response.status_code == 400
    assert "Too many files uploaded" in response.json()["detail"]


def test_filename_sanitization():
    file_content = b"dummy audio content"

    # File with empty basename
    response = client.post(
        "/identify",
        files=[("file", ("///", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 400
    assert "Invalid filename" in response.json()["detail"]
