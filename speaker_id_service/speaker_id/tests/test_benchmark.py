import os
import tempfile
import io
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers
import asyncio

from benchmark import (
    create_large_file,
    simulate_concurrent_requests_blocking,
    simulate_concurrent_requests_nonblocking,
    measure_event_loop_lag
)

def test_create_large_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dummy.txt")
        create_large_file(path, size_mb=1)
        assert os.path.exists(path)
        assert os.stat(path).st_size == 1 * 1024 * 1024

@pytest.mark.asyncio
async def test_simulate_concurrent_requests_blocking():
    files = []
    for _ in range(3):
        f = io.BytesIO(b"test data")
        upload_file = UploadFile(filename="test.wav", file=f, headers=Headers())
        files.append(upload_file)

    duration = await simulate_concurrent_requests_blocking(files)
    assert duration >= 0
    # Files should be seeked back to 0
    for f in files:
        assert f.file.tell() == 0

@pytest.mark.asyncio
async def test_simulate_concurrent_requests_nonblocking():
    files = []
    for _ in range(3):
        f = io.BytesIO(b"test data")
        upload_file = UploadFile(filename="test.wav", file=f, headers=Headers())
        files.append(upload_file)

    duration = await simulate_concurrent_requests_nonblocking(files)
    assert duration >= 0
    # Files should be seeked back to 0
    for f in files:
        assert f.file.tell() == 0

@pytest.mark.asyncio
async def test_measure_event_loop_lag():
    async def dummy_coro(files):
        await asyncio.sleep(0.05)
        return 0.5

    duration, max_delay = await measure_event_loop_lag(dummy_coro, [])
    assert duration == 0.5
    assert max_delay >= 0
