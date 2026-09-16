import pytest
import io
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys

# Isolate the mock from test_app
class MockTorchRL(MagicMock):
    pass

sys.modules['torch'] = MockTorchRL()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.compliance'] = MagicMock()
sys.modules['torchaudio.compliance.kaldi'] = MagicMock()
sys.modules['campplus_model'] = MagicMock()

import urllib.request
_orig_urlretrieve = urllib.request.urlretrieve
urllib.request.urlretrieve = MagicMock()
import hashlib
_orig_hashlib = hashlib.sha256
_mock_sha256 = MagicMock()
_mock_sha256.return_value.hexdigest.return_value = "07abeeb5150441995b51ea65c9ccc8feed78b33040012f1d2fad29a0e4f5b8d7"
hashlib.sha256 = _mock_sha256

import builtins
_orig_open = builtins.open
builtins.open = MagicMock()
builtins.open.return_value.__enter__.return_value.read.side_effect = [b"", b""]

import torch
torch.load = MagicMock(return_value={})
torch.cuda = MagicMock()
torch.cuda.is_available.return_value = False

import app

client = TestClient(app.app)

def test_rate_limiting():
    """Test that requests are rate limited after hitting the max limit."""
    file_content = b"dummy audio content"

    # Send requests to hit the limit
    for _ in range(app.RATE_LIMIT_MAX_REQUESTS):
        response = client.post(
            "/identify",
            files=[("file", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
        )
        assert response.status_code != 429

    # The next one should be rate limited
    response = client.post(
        "/identify",
        files=[("file", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 429
    assert "Too Many Requests" in response.json()["detail"]

    # Test proxy header
    # Clean up rate limits for tests
    app._rate_limits.clear()
    for _ in range(app.RATE_LIMIT_MAX_REQUESTS):
        response = client.post(
            "/identify",
            headers={"X-Forwarded-For": "192.168.1.100"},
            files=[("file", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
        )
        assert response.status_code != 429

    response = client.post(
        "/identify",
        headers={"X-Forwarded-For": "192.168.1.100"},
        files=[("file", ("test.wav", io.BytesIO(file_content), "audio/wav"))]
    )
    assert response.status_code == 429
