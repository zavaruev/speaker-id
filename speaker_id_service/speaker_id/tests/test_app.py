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

# Restore real modules once app is imported, so that subsequent tests
# (model/pooling layers) run against the actual torch/torchaudio.
for _mod in ('torch', 'torch.nn', 'torch.nn.functional', 'torchaudio',
             'torchaudio.compliance', 'torchaudio.compliance.kaldi', 'campplus_model'):
    sys.modules.pop(_mod, None)
urllib.request.urlretrieve = _orig_urlretrieve
hashlib.sha256 = _orig_hashlib
builtins.open = _orig_open

from fastapi.testclient import TestClient

client = TestClient(app.app)

from unittest.mock import AsyncMock

from fastapi import HTTPException

@pytest.mark.asyncio
async def test_get_api_key_valid_default():
    """Test get_api_key raises 401 when API_KEY is unset (no hardcoded fallback since PR #79)"""
    with patch.dict(os.environ, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await app.get_api_key("default_secret_key")
        assert exc_info.value.status_code == 401

@pytest.mark.asyncio
async def test_get_api_key_valid_custom():
    """Test get_api_key returns the key when it matches a custom API_KEY"""
    with patch.dict(os.environ, {"API_KEY": "my_custom_secret_key"}):
        result = await app.get_api_key("my_custom_secret_key")
        assert result == "my_custom_secret_key"

@pytest.mark.asyncio
async def test_get_api_key_invalid():
    """Test get_api_key raises HTTPException when API key is invalid"""
    with patch.dict(os.environ, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await app.get_api_key("invalid_key")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or missing API Key"

@pytest.mark.asyncio
@patch("app.asyncio.create_subprocess_exec")
async def test_convert_to_wav_success(mock_create_subprocess_exec):
    """Test successful conversion returns True"""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock()
    mock_process.returncode = 0
    mock_create_subprocess_exec.return_value = mock_process

    result = await app.convert_to_wav("input.wav", "output.wav")
    assert result is True

@pytest.mark.asyncio
@patch("app.asyncio.create_subprocess_exec")
async def test_convert_to_wav_called_process_error(mock_create_subprocess_exec):
    """Test that non-zero return code returns False"""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock()
    mock_process.returncode = 1
    mock_create_subprocess_exec.return_value = mock_process

    result = await app.convert_to_wav("input.wav", "output.wav")
    assert result is False

@pytest.mark.asyncio
@patch("app.asyncio.create_subprocess_exec")
async def test_convert_to_wav_shell_injection(mock_create_subprocess_exec):
    """Test that command injection is prevented by string casting and passing args directly to exec"""
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock()
    mock_process.returncode = 0
    mock_create_subprocess_exec.return_value = mock_process

    await app.convert_to_wav("input.wav", "-ar 8000; rm -rf /")
    mock_create_subprocess_exec.assert_called_once_with(
        'ffmpeg', '-y', '-i', 'input.wav',
        '-ar', '16000', '-ac', '1', '-ar 8000; rm -rf /',
        stdout=app.asyncio.subprocess.DEVNULL,
        stderr=app.asyncio.subprocess.DEVNULL
    )

@pytest.mark.asyncio
@patch("app.asyncio.create_subprocess_exec")
async def test_convert_to_wav_file_not_found(mock_create_subprocess_exec):
    """Test that missing ffmpeg binary returns False"""
    mock_create_subprocess_exec.side_effect = FileNotFoundError()

    result = await app.convert_to_wav("input.wav", "output.wav")
    assert result is False

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

@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav", new_callable=AsyncMock, return_value=True)
@patch("app.torchaudio.load")
@patch("os.path.getsize", return_value=1024)
@patch("os.remove", return_value=None)
def test_identify_empty_signal(mock_remove, mock_getsize, mock_load, mock_convert, mock_open):
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


def test_enroll_form_success():
    """Test that the /enroll HTML form returns correctly."""
    response = client.get("/enroll")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert b"<!DOCTYPE html>" in response.content
    assert b"<title>Voice Enrollment | Speaker ID</title>" in response.content
    assert b"form-group" in response.content

@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav", new_callable=AsyncMock)
@patch("app.torchaudio.load")
@patch("os.path.getsize", return_value=1024)
@patch("os.remove", return_value=None)
def test_identify_path_traversal(mock_remove, mock_getsize, mock_load, mock_convert, mock_open):
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

@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav", new_callable=AsyncMock)
@patch("app.torchaudio.load")
@patch("os.path.getsize", return_value=1024)
@patch("os.remove", return_value=None)
def test_enroll_path_traversal(mock_remove, mock_getsize, mock_load, mock_convert, mock_open):
    """Test that path traversal in filenames is prevented in enroll"""
    mock_convert.return_value = False # fail early to avoid ML pipeline

    malicious_filename = "../../../etc/shadow"
    response = client.post(
        "/enroll",
        headers={"X-API-Key": "default_secret_key"},
        data={"user_id": "test_user"},
        files=[("files", (malicious_filename, b"dummy content", "audio/mpeg"))]
    )

    # Check open calls
    open_args = mock_open.call_args[0][0]
    assert open_args.startswith("/tmp/")
    assert "../" not in open_args
    assert "/etc/" not in open_args

@patch("builtins.open", new_callable=MagicMock)
@patch("app.convert_to_wav", new_callable=AsyncMock, return_value=True)
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
        headers={"X-API-Key": "default_secret_key"},
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
@patch("app.convert_to_wav", new_callable=AsyncMock)
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


@patch("app.kaldi.fbank")
def test_compute_fbank_16k(mock_fbank):
    """Test compute_fbank when sample rate is exactly 16000Hz."""
    mock_signal = MagicMock()
    mock_fbank_tensor = MagicMock()

    # Setup for fbank - fbank.mean() and fbank.unsqueeze(0)
    mock_mean_tensor = MagicMock()
    mock_sub_tensor = MagicMock()
    mock_unsqueeze_tensor = MagicMock()

    mock_fbank_tensor.mean.return_value = mock_mean_tensor
    mock_fbank_tensor.__sub__.return_value = mock_sub_tensor
    mock_sub_tensor.unsqueeze.return_value = mock_unsqueeze_tensor

    mock_fbank.return_value = mock_fbank_tensor

    result = app.compute_fbank(mock_signal, 16000)

    # Assert kaldi fbank was called with 16000 fs and original signal
    mock_fbank.assert_called_once_with(
        mock_signal, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=16000
    )

    # Check tensor operations
    mock_fbank_tensor.mean.assert_called_once_with(dim=0, keepdim=True)
    mock_fbank_tensor.__sub__.assert_called_once_with(mock_mean_tensor)
    mock_sub_tensor.unsqueeze.assert_called_once_with(0)

    assert result == mock_unsqueeze_tensor


@patch("app.kaldi.fbank")
@patch("app.torchaudio.transforms.Resample")
def test_compute_fbank_not_16k(mock_resample_class, mock_fbank):
    """Test compute_fbank when sample rate is not 16000Hz and needs resampling."""
    # Let's ensure _resampler_16k has a specific mock so we can test the behavior
    # when its frequencies don't match.
    original_resampler = getattr(app, '_resampler_16k', None)
    dummy_resampler = MagicMock()
    dummy_resampler.orig_freq = 0
    dummy_resampler.new_freq = 0
    app._resampler_16k = dummy_resampler

    mock_signal = MagicMock()
    mock_signal.device = "cuda:0"

    # Mock the resampler instance and its .to() method
    mock_resampler_instance = MagicMock()
    mock_resampler_instance.orig_freq = 8000
    mock_resampler_instance.new_freq = 16000
    mock_resampler_instance_to = MagicMock()
    mock_resampler_instance.to.return_value = mock_resampler_instance_to

    # Calling the resampler on the signal returns a resampled signal
    mock_resampled_signal = MagicMock()
    mock_resampler_instance_to.return_value = mock_resampled_signal

    mock_resample_class.return_value = mock_resampler_instance

    # Mock fbank processing similar to the 16k test
    mock_fbank_tensor = MagicMock()
    mock_mean_tensor = MagicMock()
    mock_sub_tensor = MagicMock()
    mock_unsqueeze_tensor = MagicMock()

    mock_fbank_tensor.mean.return_value = mock_mean_tensor
    mock_fbank_tensor.__sub__.return_value = mock_sub_tensor
    mock_sub_tensor.unsqueeze.return_value = mock_unsqueeze_tensor

    mock_fbank.return_value = mock_fbank_tensor

    try:
        # Initializing _resampler_16k case
        result = app.compute_fbank(mock_signal, 8000)

        # Verify Resample initialized and moved to device
        mock_resample_class.assert_called_once_with(8000, 16000)
        mock_resampler_instance.to.assert_called_once_with("cuda:0")

        # Verify resampler was called with original signal
        mock_resampler_instance_to.assert_called_once_with(mock_signal)

        # Verify kaldi fbank called with resampled signal and 16000 fs
        mock_fbank.assert_called_once_with(
            mock_resampled_signal, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=16000
        )

        assert result == mock_unsqueeze_tensor
        assert hasattr(app, '_resampler_16k')

        # For testing reuse, we must ensure the mock that was placed into app._resampler_16k
        # by the compute_fbank function (which is mock_resampler_instance_to) has the right properties
        # so it doesn't trigger a re-initialization.
        mock_resampler_instance_to.orig_freq = 8000
        mock_resampler_instance_to.new_freq = 16000

        # Test reuse of resampler
        mock_resample_class.reset_mock()
        mock_fbank.reset_mock()
        mock_resampler_instance_to.reset_mock()

        result2 = app.compute_fbank(mock_signal, 8000)

        mock_resample_class.assert_not_called()  # Resampler class shouldn't be instanciated again
        mock_resampler_instance_to.assert_called_once_with(mock_signal)
        mock_fbank.assert_called_once_with(
            mock_resampled_signal, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=16000
        )

        # Test re-initialization of resampler if fs changes
        mock_resample_class.reset_mock()
        mock_resample_class.return_value = mock_resampler_instance

        result3 = app.compute_fbank(mock_signal, 24000)
        mock_resample_class.assert_called_once_with(24000, 16000)

    finally:
        if original_resampler is not None:
            app._resampler_16k = original_resampler


@patch("os.remove")
def test_safe_remove_success(mock_remove):
    """Test successful file removal."""
    app._safe_remove("test_path.txt")
    mock_remove.assert_called_once_with("test_path.txt")


@patch("os.remove")
def test_safe_remove_file_not_found(mock_remove):
    """Test that FileNotFoundError is gracefully ignored."""
    mock_remove.side_effect = FileNotFoundError()
    # The function should not raise an exception
    app._safe_remove("test_path.txt")
    mock_remove.assert_called_once_with("test_path.txt")


@patch("os.remove")
@patch("app.logger.warning")
def test_safe_remove_exception(mock_logger_warning, mock_remove):
    """Test that other exceptions are caught and logged."""
    error_msg = "Permission denied"
    mock_remove.side_effect = Exception(error_msg)

    app._safe_remove("test_path.txt")

    mock_remove.assert_called_once_with("test_path.txt")
    mock_logger_warning.assert_called_once_with(f"Failed to remove test_path.txt: {error_msg}")
