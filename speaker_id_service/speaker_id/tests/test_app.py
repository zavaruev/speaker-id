import pytest
from unittest.mock import patch, MagicMock
import os
import shutil

# Mock ML operations before importing app
with patch('urllib.request.urlretrieve'), \
     patch('torch.load', return_value={}), \
     patch('campplus_model.CAMPPlus.load_state_dict'), \
     patch('campplus_model.CAMPPlus.to'), \
     patch('campplus_model.CAMPPlus.eval'):
    from app import app

from fastapi.testclient import TestClient

client = TestClient(app)

@patch('app.convert_to_wav')
@patch('app.torchaudio.load')
@patch('app.compute_fbank')
@patch('app.model')
@patch('app.np.load')
def test_identify_path_traversal(mock_np_load, mock_model, mock_fbank, mock_torchaudio_load, mock_convert):
    # Mocking necessary parts so the endpoint doesn't fail before testing file path
    mock_convert.return_value = True
    import torch
    mock_signal = torch.zeros(1, 8000)
    mock_torchaudio_load.return_value = (mock_signal, 16000)
    # Return empty signal to fail early or mock it properly
    # Actually, we can just let it fail on convert_to_wav or signal validation
    # Our goal is to ensure the temp file created doesn't have path traversal characters.

    # We will patch builtins.open to intercept the filename
    with patch('builtins.open') as mock_open:
        # Mock file writing to avoid errors
        mock_open.return_value.__enter__.return_value = MagicMock()

        test_file_content = b"fake audio data"
        files = {"file": ("../../../etc/passwd", test_file_content, "audio/wav")}

        response = client.post("/identify", files=files)

        # We don't care if it returns 500 or 400 because of mocked parts.
        # We just want to check the open() call

        # Find the call to open() for the temp file
        open_calls = mock_open.call_args_list
        found_temp_file = False
        for call in open_calls:
            filename = call[0][0]
            if str(filename).startswith('/tmp/'):
                found_temp_file = True
                assert "../../../etc/passwd" not in str(filename)
                assert "passwd" in str(filename)

        assert found_temp_file, "Temp file was not created"

@patch('app.convert_to_wav')
@patch('app.torchaudio.load')
@patch('app.compute_fbank')
@patch('app.model')
@patch('app.np.save')
def test_enroll_path_traversal(mock_np_save, mock_model, mock_fbank, mock_torchaudio_load, mock_convert):
    mock_convert.return_value = True
    import torch
    mock_signal = torch.zeros(1, 8000)
    mock_torchaudio_load.return_value = (mock_signal, 16000)

    import torch
    mock_model.return_value = torch.zeros(1, 512)
    with patch('builtins.open') as mock_open:
        mock_open.return_value.__enter__.return_value = MagicMock()

        test_file_content = b"fake audio data"
        files = [("files", ("../../../etc/shadow", test_file_content, "audio/wav"))]
        data = {"user_id": "test_user"}

        response = client.post("/enroll", files=files, data=data)

        open_calls = mock_open.call_args_list
        found_temp_file = False
        for call in open_calls:
            filename = call[0][0]
            if str(filename).startswith('/tmp/'):
                found_temp_file = True
                assert "../../../etc/shadow" not in str(filename)
                assert "shadow" in str(filename)

        assert found_temp_file, "Temp file was not created"
