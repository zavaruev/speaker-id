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
)

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
