import os
import io
import tempfile
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers
from benchmark import (
    create_large_file,
    simulate_concurrent_requests_blocking,
    simulate_concurrent_requests_nonblocking,
    measure_event_loop_lag,
    main,
)
from unittest.mock import patch

def test_create_large_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dummy.txt")
        create_large_file(path, size_mb=1)
        assert os.path.exists(path)
        assert os.stat(path).st_size == 1 * 1024 * 1024

@pytest.fixture
def dummy_upload_files():
    files = []
    for _ in range(3):
        f = io.BytesIO(b"dummy audio data")
        upload_file = UploadFile(filename="test.wav", file=f, headers=Headers())
        files.append(upload_file)
    return files

@pytest.mark.asyncio
async def test_simulate_concurrent_requests_blocking(dummy_upload_files):
    duration = await simulate_concurrent_requests_blocking(dummy_upload_files)
    assert duration >= 0.0

@pytest.mark.asyncio
async def test_simulate_concurrent_requests_nonblocking(dummy_upload_files):
    duration = await simulate_concurrent_requests_nonblocking(dummy_upload_files)
    assert duration >= 0.0

@pytest.mark.asyncio
async def test_measure_event_loop_lag(dummy_upload_files):
    duration, max_delay = await measure_event_loop_lag(
        simulate_concurrent_requests_nonblocking, dummy_upload_files
    )
    assert duration >= 0.0
    assert max_delay >= 0.0

@pytest.mark.asyncio
async def test_main(capsys):
    from unittest.mock import mock_open

    with patch('benchmark.create_large_file') as mock_create, \
         patch('benchmark.measure_event_loop_lag', return_value=(0.1, 0.01)) as mock_measure, \
         patch('builtins.open', mock_open(read_data=b"dummy audio data")), \
         patch('os.remove') as mock_remove:

        await main()

        # Verify print statement
        captured = capsys.readouterr()
        assert "Preparing test files..." in captured.out
        assert "Simulating 10 concurrent requests" in captured.out
        assert "Blocking method:" in captured.out
        assert "Non-blocking method:" in captured.out

        # Verify create_large_file call
        mock_create.assert_called_once_with("dummy_audio.wav", size_mb=20)

        # Verify measure_event_loop_lag calls (3 times: non-blocking warmup, blocking, non-blocking)
        assert mock_measure.call_count == 3

        # Verify cleanup
        mock_remove.assert_called_once_with("dummy_audio.wav")
