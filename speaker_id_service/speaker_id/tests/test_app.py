import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# We need to add the app directory to sys.path so that we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock heavy modules before app import
class MockTorch(MagicMock):
    pass

sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.compliance'] = MagicMock()
sys.modules['torchaudio.compliance.kaldi'] = MagicMock()
sys.modules['campplus_model'] = MagicMock()

import urllib.request
urllib.request.urlretrieve = MagicMock()

import torch
torch.load = MagicMock(return_value={})
torch.cuda = MagicMock()
torch.cuda.is_available.return_value = False

import app
from fastapi.testclient import TestClient

client = TestClient(app.app)

@patch("app.subprocess.run")
def test_convert_to_wav_shell_injection(mock_subprocess_run):
    """Test that command injection is prevented by shell=False and string casting"""
    app.convert_to_wav("input.wav", "-ar 8000; rm -rf /")
    mock_subprocess_run.assert_called_once_with([
        'ffmpeg', '-y', '-i', 'input.wav',
        '-ar', '16000', '-ac', '1', '-ar 8000; rm -rf /'
    ], check=True, stdout=app.subprocess.DEVNULL, stderr=app.subprocess.DEVNULL, shell=False)

@patch("app.shutil.copyfileobj")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav")
@patch("app.torchaudio.load")
def test_identify_path_traversal(mock_load, mock_convert, mock_open, mock_copy):
    """Test that path traversal in filenames is prevented"""
    mock_convert.return_value = False # fail early to avoid ML pipeline

    malicious_filename = "../../../etc/passwd"
    response = client.post(
        "/identify",
        files={"file": (malicious_filename, b"dummy content", "audio/mpeg")}
    )

    # Assert that open was called with a safe path that doesn't include the traversal
    # open should be called with /tmp/{uuid}_passwd
    open_args = mock_open.call_args[0][0]
    assert open_args.startswith("/tmp/")
    assert "passwd" in open_args
    assert "../" not in open_args
    assert "/etc/" not in open_args

@patch("app.shutil.copyfileobj")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav")
@patch("app.torchaudio.load")
def test_enroll_path_traversal(mock_load, mock_convert, mock_open, mock_copy):
    """Test that path traversal in filenames is prevented in enroll"""
    mock_convert.return_value = False # fail early to avoid ML pipeline

    malicious_filename = "../../../etc/shadow"
    response = client.post(
        "/enroll",
        data={"user_id": "test_user"},
        files=[("files", (malicious_filename, b"dummy content", "audio/mpeg"))]
    )

    # Check open calls
    open_args = mock_open.call_args[0][0]
    assert open_args.startswith("/tmp/")
    assert "shadow" in open_args
    assert "../" not in open_args
    assert "/etc/" not in open_args
