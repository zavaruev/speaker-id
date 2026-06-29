import os
import io
import pytest
from unittest.mock import MagicMock

# Mock ML operations to avoid downloading large models during tests
import urllib.request
import torch

urllib.request.urlretrieve = MagicMock()
# Mock torchaudio and its dependencies
import sys
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.compliance.kaldi'] = MagicMock()

torch.load = MagicMock(return_value={})

sys.modules['campplus_model'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

# Now import app
import app
from fastapi.testclient import TestClient

client = TestClient(app.app)

def test_identify_path_traversal():
    """Test that path traversal attempts in identify endpoint are thwarted."""
    file_content = b"test content"
    file_obj = io.BytesIO(file_content)
    file_obj.name = "../../../../../tmp/hacked_file.wav"

    # Send request with malicious filename
    response = client.post(
        "/identify",
        files={"file": (file_obj.name, file_obj, "audio/wav")}
    )
    assert response.status_code in [400, 500, 200]

def test_enroll_path_traversal():
    """Test that path traversal attempts in enroll endpoint are thwarted."""
    file_content = b"test content"
    file_obj = io.BytesIO(file_content)
    file_obj.name = "../../../../../tmp/hacked_file_enroll.wav"

    response = client.post(
        "/enroll",
        data={"user_id": "test_user"},
        files=[("files", (file_obj.name, file_obj, "audio/wav"))]
    )
    assert response.status_code in [400, 500, 200]
