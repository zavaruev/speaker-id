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
def test_convert_to_wav_success(mock_subprocess_run):
    """Test successful conversion returns True"""
    mock_subprocess_run.return_value = MagicMock()
    result = app.convert_to_wav("input.wav", "output.wav")
    assert result is True

@patch("app.subprocess.run")
def test_convert_to_wav_called_process_error(mock_subprocess_run):
    """Test subprocess.CalledProcessError is caught and returns False"""
    import subprocess
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, 'ffmpeg')
    result = app.convert_to_wav("input.wav", "output.wav")
    assert result is False

@patch("app.subprocess.run")
def test_convert_to_wav_shell_injection(mock_subprocess_run):
    """Test that command injection is prevented by shell=False and string casting"""
    app.convert_to_wav("input.wav", "-ar 8000; rm -rf /")
    mock_subprocess_run.assert_called_once_with([
        'ffmpeg', '-y', '-i', 'input.wav',
        '-ar', '16000', '-ac', '1', '-ar 8000; rm -rf /'
    ], check=True, stdout=app.subprocess.DEVNULL, stderr=app.subprocess.DEVNULL, shell=False)

def test_health_ready():
    """Test health endpoint when model is ready"""
    # By default in the mock setup, app._model_ready is True or we can explicitly set it
    original_ready = app._model_ready
    try:
        app._model_ready = True
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        app._model_ready = original_ready

def test_health_not_ready():
    """Test health endpoint when model is not ready"""
    original_ready = app._model_ready
    try:
        app._model_ready = False
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"detail": "Model not ready"}
    finally:
        app._model_ready = original_ready

@patch("app.shutil.copyfileobj")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav", return_value=True)
@patch("app.torchaudio.load")
@patch("os.path.getsize", return_value=1024)
@patch("os.remove", return_value=None)
def test_identify_empty_signal(mock_remove, mock_getsize, mock_load, mock_convert, mock_open, mock_copy):
    """Test identify endpoint handles empty/short audio signal"""

    # Mocking torch tensor shape since torch itself is mocked in this file
    mock_tensor = MagicMock()
    mock_tensor.numel.return_value = 0
    mock_tensor.shape = [0, 0]

    mock_load.return_value = (mock_tensor, 16000)

    response = client.post(
        "/identify",
        files={"file": ("test.wav", b"dummy content", "audio/wav")}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Audio too short or empty"}


@patch("app.shutil.copyfileobj")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav")
@patch("app.torchaudio.load")
@patch("os.path.getsize", return_value=1024)
@patch("os.remove", return_value=None)
def test_identify_path_traversal(mock_remove, mock_getsize, mock_load, mock_convert, mock_open, mock_copy):
    """Test that path traversal in filenames is prevented"""
    mock_convert.return_value = False # fail early to avoid ML pipeline

    malicious_filename = "../../../etc/passwd"
    response = client.post(
        "/identify",
        files={"file": (malicious_filename, b"dummy content", "audio/mpeg")}
    )

    # Assert that open was called with a safe path that doesn't include the traversal
    open_args = mock_open.call_args[0][0]
    assert open_args.startswith("/tmp/")
    assert "../" not in open_args
    assert "/etc/" not in open_args

@patch("app.shutil.copyfileobj")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav")
@patch("app.torchaudio.load")
@patch("os.path.getsize", return_value=1024)
@patch("os.remove", return_value=None)
def test_enroll_path_traversal(mock_remove, mock_getsize, mock_load, mock_convert, mock_open, mock_copy):
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
    assert "../" not in open_args
    assert "/etc/" not in open_args

@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav", return_value=True)
@patch("app.torchaudio.load")
@patch("os.remove", return_value=None)
def test_enroll_empty_signal(mock_remove, mock_load, mock_convert, mock_open):
    """Test that an empty or too short signal in /enroll raises a 400 error."""
    mock_signal = MagicMock()
    mock_signal.numel.return_value = 1000
    mock_signal.shape = [1, 1000]
    mock_load.return_value = (mock_signal, 16000)

    response = client.post(
        "/enroll",
        data={"user_id": "test_user"},
        files=[("files", ("test.wav", b"dummy content", "audio/wav"))]
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio too short or empty"

@patch("app.Path.glob")
def test_rebuild_cache_empty(mock_glob):
    """Test _rebuild_cache when no speaker files are present"""
    mock_glob.return_value = []

    app._rebuild_cache()

    assert app._embedding_names == []
    assert app._embedding_matrix is None

@patch("app.Path.glob")
@patch("app.np.load")
@patch("app.torch.tensor")
@patch("app.F.normalize")
@patch("app.torch.stack")
def test_rebuild_cache_success(mock_stack, mock_normalize, mock_tensor, mock_load, mock_glob):
    """Test _rebuild_cache successfully loading valid speaker files"""
    class MockPath:
        def __init__(self, name):
            self.name = name
            self.stem = name.split(".")[0]
        def __lt__(self, other):
            return self.name < other.name

    mock_file1 = MockPath("user1.npy")
    mock_file2 = MockPath("user2.npy")
    mock_glob.return_value = [mock_file1, mock_file2]

    mock_load.return_value = "mock_np_array"

    mock_tensor_obj1 = MagicMock()
    mock_tensor_obj2 = MagicMock()
    mock_tensor.side_effect = [mock_tensor_obj1, mock_tensor_obj2]

    mock_norm_obj1 = MagicMock()
    mock_norm_obj2 = MagicMock()
    mock_normalize.side_effect = [mock_norm_obj1, mock_norm_obj2]

    mock_stacked = MagicMock()
    mock_stack.return_value = mock_stacked

    app._rebuild_cache()

    assert app._embedding_names == ["user1", "user2"]
    assert app._embedding_matrix is mock_stacked

    assert mock_load.call_count == 2
    assert mock_tensor.call_count == 2
    assert mock_normalize.call_count == 2
    mock_stack.assert_called_once_with([mock_norm_obj1, mock_norm_obj2])

@patch("app.Path.glob")
@patch("app.np.load")
@patch("app.torch.tensor")
@patch("app.F.normalize")
@patch("app.torch.stack")
def test_rebuild_cache_partial_failure(mock_stack, mock_normalize, mock_tensor, mock_load, mock_glob):
    """Test _rebuild_cache skipping corrupted files and loading valid ones"""
    class MockPath:
        def __init__(self, name):
            self.name = name
            self.stem = name.split(".")[0]
        def __lt__(self, other):
            return self.name < other.name

    mock_file_corrupt = MockPath("corrupt_user.npy")
    mock_file_valid = MockPath("valid_user.npy")

    mock_glob.return_value = [mock_file_corrupt, mock_file_valid]

    # First call raises exception, second succeeds
    mock_load.side_effect = [Exception("Corrupted file"), "mock_np_array"]

    mock_tensor_obj = MagicMock()
    mock_tensor.return_value = mock_tensor_obj

    mock_norm_obj = MagicMock()
    mock_normalize.return_value = mock_norm_obj

    mock_stacked = MagicMock()
    mock_stack.return_value = mock_stacked

    app._rebuild_cache()

    assert app._embedding_names == ["valid_user"]
    assert app._embedding_matrix is mock_stacked

    # We only call tensor and normalize once because the first loop iteration fails at np.load
    assert mock_tensor.call_count == 1
    assert mock_normalize.call_count == 1
    mock_stack.assert_called_once_with([mock_norm_obj])

@patch("app.MAX_FILE_SIZE", 1024)
def test_identify_file_size_limit():
    """Test that uploading a file larger than MAX_FILE_SIZE returns 413"""
    # Generate a payload larger than the patched limit
    large_content = b"0" * 2048

    response = client.post(
        "/identify",
        files={"file": ("large_file.wav", large_content, "audio/wav")}
    )

    assert response.status_code == 413
    assert "File exceeds" in response.json()["detail"]

@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav")
@patch("app.torchaudio.load")
@patch("os.remove", return_value=None)
@patch("app.compute_fbank")
@patch("app.F.normalize")
@patch("app.logger")
def test_identify_gpu_fallback(mock_logger, mock_normalize, mock_fbank, mock_remove, mock_load, mock_convert, mock_open):
    """Test that GPU fallback executes when model(fbank) raises RuntimeError."""
    mock_convert.return_value = True

    mock_signal = MagicMock()
    mock_signal.numel.return_value = 8000
    mock_signal.shape = [1, 8000]
    mock_signal.abs().max.return_value = 1.0
    mock_signal.to.return_value = mock_signal
    mock_load.return_value = (mock_signal, 16000)

    mock_fbank_tensor = MagicMock()
    mock_fbank_tensor_cpu = MagicMock()
    mock_fbank_tensor.cpu.return_value = mock_fbank_tensor_cpu
    mock_fbank.return_value = mock_fbank_tensor

    mock_cpu_model = MagicMock()
    mock_cpu_model_embedding = MagicMock()
    mock_cpu_model_embedding.to.return_value = MagicMock()
    mock_cpu_model.return_value = mock_cpu_model_embedding

    with patch("app.model") as mock_model, \
         patch("app._embedding_matrix", None), \
         patch("app._rebuild_cache"):

        mock_model.side_effect = RuntimeError("OOM")
        mock_model.cpu.return_value = mock_cpu_model

        response = client.post(
            "/identify",
            files={"file": ("test.wav", b"dummy content", "audio/wav")}
        )

        assert response.status_code == 200
        mock_model.cpu.assert_called_once()
        mock_fbank_tensor.cpu.assert_called_once()
        mock_cpu_model.assert_called_once_with(mock_fbank_tensor_cpu)
        mock_model.to.assert_called_once_with(app.device)
        mock_logger.warning.assert_called_with("GPU inference failed, falling back to CPU: OOM")
